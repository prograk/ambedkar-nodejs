/* eslint-disable no-undef */
import express from 'express';
import cors from 'cors';
import multer from 'multer';
import { QdrantClient } from '@qdrant/js-client-rest';
import { pipeline, env } from '@xenova/transformers';
import { config } from 'dotenv';
import { randomUUID } from 'crypto';
import { encoding_for_model } from 'tiktoken';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';

config();

const app = express();
const PORT = process.env.PORT || 3001;

// Configure multer for file uploads with streaming
const upload = multer({ 
  storage: multer.memoryStorage(),
  limits: { fileSize: 10 * 1024 * 1024 }, // Reduced to 10MB limit
  fileFilter: (req, file, cb) => {
    if (file.mimetype === 'application/pdf') {
      cb(null, true);
    } else {
      cb(new Error('Only PDF files are allowed'), false);
    }
  }
});

// Middleware
if (process.env.NODE_ENV === 'development') {
  // Allow all origins in development
  app.use(cors());
  console.log('🔓 CORS disabled for development');
} else {
  // Strict CORS in production
  app.use(cors({
    origin: process.env.FRONTEND_URL || 'http://localhost:3000',
    credentials: true
  }));
  console.log('🔒 CORS enabled for production');
}
app.use(express.json({ limit: '10mb' }));

// Configuration
const QDRANT_CONFIG = {
  url: process.env.QDRANT_URL,
  apiKey: process.env.QDRANT_API_KEY,
  collectionName: process.env.QDRANT_COLLECTION_NAME
};

const CHUNK_CONFIG = {
  chunkSize: 800, // Reduced chunk size
  chunkOverlap: 100, // Reduced overlap
  maxChunks: 50, // Reduced max chunks to prevent memory overload
  separators: ["\n\n", "\n", "।", ".", "?", "!"],
};

// Global instances
let qdrantClient = null;
let embeddingModel = null;
let isModelLoading = false;
let documentCache = [];
let cacheLastUpdated = null;
let modelLastUsed = null;
const MODEL_IDLE_TIMEOUT = 5 * 60 * 1000; // 5 minutes

// Initialize services
const initializeServices = async () => {
  try {
    // Initialize Qdrant
    if (!QDRANT_CONFIG.url || !QDRANT_CONFIG.apiKey) {
      throw new Error('Qdrant configuration missing');
    }
    
    qdrantClient = new QdrantClient({
      url: QDRANT_CONFIG.url,
      apiKey: QDRANT_CONFIG.apiKey,
    });
    
    console.log('✅ Qdrant client initialized');

    await VectorService.createPayloadIndexes();
    
    // Don't pre-load embedding model - load on demand
    await initializeEmbeddingModel();
    
    await VectorService.refreshDocumentCache();

    return true;
  } catch (error) {
    console.error('❌ Service initialization failed:', error);
    throw error;
  }
};

const initializeEmbeddingModel = async () => {
  if (embeddingModel) {
    modelLastUsed = Date.now();
    return embeddingModel;
  }
  
  if (isModelLoading) {
    while (isModelLoading) {
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    return embeddingModel;
  }
  
  try {
    isModelLoading = true;
    console.log('🔄 Loading embedding model...');
    
    env.allowLocalModels = false;
    env.allowRemoteModels = true;
    env.cacheDir = './models';
    
    embeddingModel = await pipeline('feature-extraction', 'Xenova/paraphrase-multilingual-MiniLM-L12-v2', {
      dtype: 'fp32',
      device: 'cpu',
      quantized: true // Use quantized model to reduce memory
    });
    
    modelLastUsed = Date.now();
    console.log('✅ Embedding model loaded');
    
    // Set up auto-unload timer
    setInterval(() => {
      if (embeddingModel && modelLastUsed && (Date.now() - modelLastUsed) > MODEL_IDLE_TIMEOUT) {
        console.log('🧹 Unloading idle embedding model to free memory');
        embeddingModel = null;
        modelLastUsed = null;
        if (global.gc) {
          global.gc();
        }
      }
    }, 60000); // Check every minute
    
    return embeddingModel;
  } catch (error) {
    console.error('❌ Failed to load embedding model:', error);
    throw error;
  } finally {
    isModelLoading = false;
  }
};

// Core Services
class DocumentService {
  // Extract text from PDF buffer
  static async extractTextFromPDF(buffer, filename) {
    try {
      return new Promise((resolve, reject) => {
        const pdfParser = new PDFParser();
        
        pdfParser.on('pdfParser_dataError', (errData) => {
          reject(new Error(`PDF parsing error: ${errData.parserError}`));
        });
        
        pdfParser.on('pdfParser_dataReady', (pdfData) => {
          try {
            // Extract text from parsed PDF data
            let text = '';
            
            if (pdfData.Pages && pdfData.Pages.length > 0) {
              pdfData.Pages.forEach(page => {
                if (page.Texts && page.Texts.length > 0) {
                  page.Texts.forEach(textItem => {
                    if (textItem.R && textItem.R.length > 0) {
                      textItem.R.forEach(r => {
                        if (r.T) {
                          text += decodeURIComponent(r.T) + ' ';
                        }
                      });
                    }
                  });
                  text += '\n';
                }
              });
            }
            
            if (!text || text.trim().length === 0) {
              reject(new Error('PDF contains no extractable text'));
              return;
            }
            
            resolve({
              text: text.trim(),
              pages: pdfData.Pages.length,
              metadata: {
                filename,
                extractedAt: new Date().toISOString(),
                characterCount: text.length
              }
            });
          } catch (parseError) {
            reject(new Error(`Failed to process PDF data: ${parseError.message}`));
          }
        });
        
        // Parse the PDF buffer
        pdfParser.parseBuffer(buffer);
      });
    } catch (error) {
      console.error('PDF extraction failed:', error);
      throw new Error(`Failed to extract text from PDF: ${error.message}`);
    }
  }
  
  // Chunk text using LangChain
  static async chunkText(text) {
    try {
      const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: CHUNK_CONFIG.chunkSize,
        chunkOverlap: CHUNK_CONFIG.chunkOverlap,
        separators: CHUNK_CONFIG.separators,
      });

      const chunks = await splitter.splitText(text);
      const validChunks = chunks.filter(chunk => chunk.trim().length > 10);
      
      if (validChunks.length > CHUNK_CONFIG.maxChunks) {
        console.warn(`Document has ${validChunks.length} chunks, limiting to ${CHUNK_CONFIG.maxChunks}`);
        return validChunks.slice(0, CHUNK_CONFIG.maxChunks);
      }
      
      return validChunks;
    } catch (error) {
      console.error('Text chunking failed:', error);
      throw new Error(`Failed to chunk text: ${error.message}`);
    }
  }
  
  // Process complete document pipeline
  static async processDocument(buffer, filename) {
    try {
      console.log(`📄 Processing document: ${filename}`);
      
      // Extract text
      const extraction = await this.extractTextFromPDF(buffer, filename);
      
      // Create chunks
      const chunks = await this.chunkText(extraction.text);
      
      // Generate document record
      const document = {
        id: randomUUID(),
        name: filename,
        status: 'processed',
        uploadedAt: new Date().toISOString(),
        metadata: {
          ...extraction.metadata,
          chunkCount: chunks.length,
          processingMethod: 'LangChain RecursiveCharacterTextSplitter'
        },
        chunks // Keep chunks internal to service
      };
      
      console.log(`✅ Document processed: ${chunks.length} chunks from ${filename}`);
      return document;
    } catch (error) {
      console.error(`Document processing failed for ${filename}:`, error);
      throw error;
    }
  }

  static getDocumentsFromCache() {
    return {
      documents: documentCache,
      lastUpdated: cacheLastUpdated,
      count: documentCache.length
    };
  };
}

class EmbeddingService {
  static async generateEmbeddings(texts) {
    try {
      const model = await initializeEmbeddingModel();
      const embeddings = [];
      
      // Process in smaller batches to reduce memory
      const batchSize = 2; // Reduced from 5
      for (let i = 0; i < texts.length; i += batchSize) {
        const batch = texts.slice(i, i + batchSize);
        
        for (const text of batch) {
          const cleanText = text.trim().replace(/\s+/g, ' ');
          if (cleanText.length === 0) {
            embeddings.push(new Array(384).fill(0));
            continue;
          }
          
          const output = await model(cleanText, { 
            pooling: 'mean', 
            normalize: true 
          });
          
          let embedding;
          if (output?.data) {
            embedding = Array.from(output.data);
          } else if (Array.isArray(output)) {
            embedding = output;
          } else {
            throw new Error('Unexpected embedding output format');
          }
          
          if (embedding.length !== 384) {
            throw new Error(`Invalid embedding dimension: ${embedding.length}`);
          }
          
          embeddings.push(embedding);
        }
        
        // Progress logging
        if (embeddings.length % 50 === 0) {
          console.log(`📊 Generated ${embeddings.length}/${texts.length} embeddings`);
        }
      }
      
      return embeddings;
    } catch (error) {
      console.error('Embedding generation failed:', error);
      throw error;
    }
  }
}

class VectorService {
  // Create necessary indexes for efficient filtering
  static async createPayloadIndexes() {
    try {
      if (!qdrantClient) {
        throw new Error('Qdrant client not initialized');
      }

      console.log('🔧 Creating payload indexes for efficient filtering...');

      // Create index for document_id (used in delete and document-specific searches)
      try {
        await qdrantClient.createPayloadIndex(QDRANT_CONFIG.collectionName, {
          field_name: 'document_id',
          field_schema: 'keyword' // Use 'keyword' for exact string matching
        });
        console.log('✅ Created index for document_id');
      } catch (error) {
        if (error.message.includes('already exists')) {
          console.log('ℹ️ Index for document_id already exists');
        } else {
          console.error('❌ Failed to create document_id index:', error.message);
        }
      }

      // Create index for document_name (used in document filtering)
      try {
        await qdrantClient.createPayloadIndex(QDRANT_CONFIG.collectionName, {
          field_name: 'document_name',
          field_schema: 'keyword'
        });
        console.log('✅ Created index for document_name');
      } catch (error) {
        if (error.message.includes('already exists')) {
          console.log('ℹ️ Index for document_name already exists');
        } else {
          console.error('❌ Failed to create document_name index:', error.message);
        }
      }

      // Create index for uploaded_at (useful for time-based filtering)
      try {
        await qdrantClient.createPayloadIndex(QDRANT_CONFIG.collectionName, {
          field_name: 'uploaded_at',
          field_schema: 'keyword'
        });
        console.log('✅ Created index for uploaded_at');
      } catch (error) {
        if (error.message.includes('already exists')) {
          console.log('ℹ️ Index for uploaded_at already exists');
        } else {
          console.error('❌ Failed to create uploaded_at index:', error.message);
        }
      }

      console.log('🎉 Payload index creation completed');
      return true;

    } catch (error) {
      console.error('❌ Failed to create payload indexes:', error);
      throw error;
    }
  }

  static async refreshDocumentCache() {
    try {
      console.log('🔄 Refreshing document cache...');
      // Only cache essential metadata, not full documents
      const documents = await this.getDocumentList();
      documentCache = documents.map(doc => ({
        id: doc.id,
        name: doc.name,
        uploadedAt: doc.uploadedAt,
        chunkCount: doc.chunkCount
      }));
      cacheLastUpdated = new Date().toISOString();
      console.log(`✅ Document cache updated: ${documentCache.length} documents`);
      return documentCache;
    } catch (error) {
      console.error('❌ Failed to refresh document cache:', error);
      throw error;
    }
  };

  static async saveDocument(document) {
    try {
      if (!qdrantClient) {
        throw new Error('Qdrant client not initialized');
      }
      
      console.log(`💾 Saving document to vector store: ${document.name}`);
      
      // Generate embeddings
      const embeddings = await EmbeddingService.generateEmbeddings(document.chunks);
      
      // Prepare points
      const points = document.chunks.map((chunk, index) => ({
        id: randomUUID(),
        vector: embeddings[index],
        payload: {
          text: chunk,
          document_id: document.id,
          document_name: document.name,
          chunk_index: index,
          uploaded_at: document.uploadedAt,
          char_count: chunk.length
        }
      }));
      
      // Upload in smaller batches to reduce memory
      const batchSize = 20; // Reduced from 100
      let totalUploaded = 0;
      
      for (let i = 0; i < points.length; i += batchSize) {
        const batch = points.slice(i, i + batchSize);
        
        await qdrantClient.upsert(QDRANT_CONFIG.collectionName, {
          wait: true,
          points: batch
        });
        
        totalUploaded += batch.length;
        console.log(`✅ Uploaded batch: ${totalUploaded}/${points.length} chunks`);
      }
      
      console.log(`🎉 Document saved: ${document.name} (${totalUploaded} chunks)`);
      return { success: true, chunkCount: totalUploaded };
    } catch (error) {
      console.error('Vector storage failed:', error);
      throw error;
    }
  }
  
  static async searchSimilar(query, limit = 10) {
    try {
      if (!qdrantClient) {
        throw new Error('Qdrant client not initialized');
      }
      
      // Generate query embedding
      const queryEmbedding = await EmbeddingService.generateEmbeddings([query]);
      
      // Search vectors
      const searchResult = await qdrantClient.search(QDRANT_CONFIG.collectionName, {
        vector: queryEmbedding[0],
        limit: Math.min(limit, 10), // Cap search results
        with_payload: true,
        score_threshold: 0.3 // Increased threshold to reduce results
      });
      
      return searchResult || [];
    } catch (error) {
      console.error('Vector search failed:', error);
      throw error;
    }
  }
  
  static async deleteDocument(documentId) {
    try {
      if (!qdrantClient) {
        throw new Error('Qdrant client not initialized');
      }
      
      await qdrantClient.delete(QDRANT_CONFIG.collectionName, {
        filter: {
          must: [{
            key: 'document_id',
            match: { value: documentId }
          }]
        }
      });
      
      console.log(`🗑️ Deleted document: ${documentId}`);
      return true;
    } catch (error) {
      console.error('Document deletion failed:', error);
      throw error;
    }
  }
  
  static async getDocumentList() {
    try {
      if (!qdrantClient) {
        throw new Error('Qdrant client not initialized');
      }
      
      const scrollResult = await qdrantClient.scroll(QDRANT_CONFIG.collectionName, {
        limit: 100, // Reduced from 1000 to prevent memory spike
        with_payload: ['document_id', 'document_name', 'uploaded_at'], // Only needed fields
        with_vector: false
      });
      
      // Group by document
      const documentsMap = {};
      (scrollResult.points || []).forEach(point => {
        const payload = point.payload;
        const docId = payload.document_id;
        
        if (!documentsMap[docId]) {
          documentsMap[docId] = {
            id: docId,
            name: payload.document_name,
            uploadedAt: payload.uploaded_at,
            chunkCount: 0
          };
        }
        documentsMap[docId].chunkCount++;
      });
      
      return Object.values(documentsMap);
    } catch (error) {
      console.error('Failed to get document list:', error);
      return [];
    }
  }
}

class UtilService {
  static cleanLLMJson(text) {
    // Remove ```json or ``` markers
    let cleaned = text.trim();
    cleaned = cleaned.replace(/```(json)?/gi, '').replace(/```/g, '').trim();

    return cleaned;
  }

  static parseLLMJson(text) {
    try {
      const cleaned = this.cleanLLMJson(text);
      return JSON.parse(cleaned);
    } catch (error) {
      console.error('❌ JSON parse error:', error.message);
      console.error('Text that failed:', text);
      return null;
    }
  }
}

// Token estimation utilities
class TokenEstimator {
  static encoderCache = new Map();
  
  static getEncoder(model = 'gpt-4') {
    // Cache encoders to avoid repeated initialization
    if (this.encoderCache.has(model)) {
      return this.encoderCache.get(model);
    }
    
    let encoder;
    try {
      // Map your OpenRouter model to tiktoken model
      if (model.includes('gpt-4')) {
        encoder = encoding_for_model('gpt-4');
      } else if (model.includes('gpt-3.5')) {
        encoder = encoding_for_model('gpt-3.5-turbo');
      } else {
        // Default to gpt-4 for unknown models
        encoder = encoding_for_model('gpt-4');
      }
      
      this.encoderCache.set(model, encoder);
      return encoder;
    } catch (error) {
      console.warn('Failed to get tiktoken encoder, falling back to estimation:', error);
      return null;
    }
  }

  // Accurate token counting using tiktoken
  static estimateTokens(text, model = process.env.OPENROUTER_MODEL) {
    if (!text || typeof text !== 'string') return 0;
    
    const encoder = this.getEncoder(model);
    if (!encoder) {
      // Fallback to your current estimation
      return Math.ceil(text.length / 4);
    }
    
    try {
      const tokens = encoder.encode(text);
      return tokens.length;
    } catch (error) {
      console.warn('Token encoding failed, using fallback:', error);
      return Math.ceil(text.length / 4);
    }
  }

  // Accurate truncation using tiktoken
  static truncateToTokenLimit(text, maxTokens, model = process.env.OPENROUTER_MODEL) {
    if (!text) return '';
    
    const encoder = this.getEncoder(model);
    if (!encoder) {
      // Fallback to your current method
      const estimatedTokens = this.estimateTokens(text);
      if (estimatedTokens <= maxTokens) return text;
      const maxChars = maxTokens * 4;
      return text.substring(0, maxChars - 100) + '...';
    }
    
    try {
      const tokens = encoder.encode(text);
      if (tokens.length <= maxTokens) return text;
      
      const truncatedTokens = tokens.slice(0, maxTokens);
      return encoder.decode(truncatedTokens);
    } catch (error) {
      console.warn('Token truncation failed, using fallback:', error);
      const maxChars = maxTokens * 4;
      return text.substring(0, maxChars - 100) + '...';
    }
  }

  // Enhanced context building with accurate token counting
  static buildContextWithTokenLimit(searchResults, maxTokens, model = process.env.OPENROUTER_MODEL) {
    let context = '';
    let currentTokens = 0;
    const usedResults = [];

    for (const result of searchResults) {
      const chunk = `[${usedResults.length + 1}] From "${result.payload.document_name}":\n${result.payload.text}\n\n`;
      const chunkTokens = this.estimateTokens(chunk, model);
      
      if (currentTokens + chunkTokens > maxTokens) {
        break;
      }
      
      context += chunk;
      currentTokens += chunkTokens;
      usedResults.push(result);
    }

    return { context, usedResults, tokenCount: currentTokens };
  }

  // Count tokens for messages array (useful for API calls)
  static estimateMessagesTokens(messages, model = process.env.OPENROUTER_MODEL) {
    let totalTokens = 0;
    
    for (const message of messages) {
      // Add tokens for message structure
      totalTokens += 4; // Base tokens per message
      
      // Add role tokens
      totalTokens += this.estimateTokens(message.role, model);
      
      // Add content tokens
      totalTokens += this.estimateTokens(message.content, model);
    }
    
    // Add base tokens for completion
    totalTokens += 2;
    
    return totalTokens;
  }

  // Cleanup method to free encoders (call on server shutdown)
  static cleanup() {
    for (const encoder of this.encoderCache.values()) {
      try {
        encoder.free();
      } catch (error) {
        console.warn('Failed to free encoder:', error);
      }
    }
    this.encoderCache.clear();
  }
}

// ENHANCED AIService with streaming support
class AIService {
  // Configuration
  static CONFIG = {
    MAX_CONTEXT_TOKENS: 8000,    // Reserve tokens for context
    MAX_RESPONSE_TOKENS: 2000,   // Reserve tokens for response
    SEARCH_LIMIT: 25,            // Reduced from 30
    MIN_SCORE_THRESHOLD: 0.3     // Higher threshold for better results
  };

  static async llmCall(messages) {
    try {
      const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${process.env.OPENROUTER_API_KEY}`,
          'Content-Type': 'application/json',
          'HTTP-Referer': process.env.FRONTEND_URL,
          'X-Title': 'Document Chat API',
        },
        body: JSON.stringify({
          model: process.env.OPENROUTER_MODEL,
          messages,
          max_tokens: 1500,
          temperature: 0.1
        })
      });

      if (!response.ok) {
        throw new Error(`OpenRouter API error: ${response.status}`);
      }

      const data = await response.json();
      const responseText = data?.choices[0]?.message?.content || '';
      return responseText;
    } catch (error) {
      console.error('OpenRouter API call failed:', error);
      throw error;
    }
  }

  // NEW: Streaming LLM call
  static async streamingLlmCall(messages, onChunk) {
    try {
      const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${process.env.OPENROUTER_API_KEY}`,
          'Content-Type': 'application/json',
          'HTTP-Referer': process.env.FRONTEND_URL,
          'X-Title': 'Document Chat API',
        },
        body: JSON.stringify({
          model: process.env.OPENROUTER_MODEL,
          messages,
          max_tokens: this.CONFIG.MAX_RESPONSE_TOKENS,
          temperature: 0.1,
          stream: true // Enable streaming
        })
      });

      if (!response.ok) {
        throw new Error(`OpenRouter API error: ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const chunk = decoder.decode(value, { stream: true });
          buffer += chunk;

          const lines = buffer.split('\n');
          buffer = lines.pop() || '';

          for (const line of lines) {
            if (line.trim()) {
              if (line.startsWith('data: ')) {
                const data = line.slice(6);
                if (data === '[DONE]') continue;

                try {
                  const parsed = JSON.parse(data);
                  const content = parsed.choices?.[0]?.delta?.content;
                  if (content) {
                    onChunk(content);
                  }
                } catch (parseError) {
                  console.warn('Failed to parse streaming chunk:', data);
                }
              }
            }
          }
        }
      } finally {
        reader.releaseLock();
      }
    } catch (error) {
      console.error('Streaming OpenRouter API call failed:', error);
      throw error;
    }
  }

  static async summarizeBatch(batchChunks, query) {
    const batchText = batchChunks
      .map((r, i) => `[${i+1}] ${r.payload.text}`)
      .join("\n\n");

    const systemPrompt = `
  You are an expert summarizer. Your job is to read document excerpts and write a *short, focused summary* of information relevant to answering this question:

  "${query}"

  Ignore unrelated text. Be concise but keep important details.`;
    
    const userPrompt = `DOCUMENT EXCERPTS:\n${batchText}`;
    
    const messages = [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: userPrompt }
    ];
    
    return await this.llmCall(messages);
  }

  // SINGLE LLM CALL - Direct approach without batching
  static async generateResponse(query, historyContext = '') {
    try {
      console.log('🔄 Starting optimized response generation...');
      
      // 1. Search for relevant content
      const searchResults = await VectorService.searchSimilar(
        query, 
        this.CONFIG.SEARCH_LIMIT
      );

      // Filter by score threshold
      const filteredResults = searchResults.filter(
        result => result.score >= this.CONFIG.MIN_SCORE_THRESHOLD
      );

      if (filteredResults.length === 0) {
        return {
          answer: "I couldn't find relevant information in your documents to answer this question.",
          sources: [],
          relevantSections: [],
          searchStrategy: 'No relevant documents found',
          confidence: "low",
          tokenUsage: { context: 0, response: 0, total: 0 }
        };
      }

      console.log(`📊 Found ${filteredResults.length} relevant chunks (score >= ${this.CONFIG.MIN_SCORE_THRESHOLD})`);

      // 2. Build context with token limits
      const { context, usedResults, tokenCount } = TokenEstimator.buildContextWithTokenLimit(
        filteredResults, 
        this.CONFIG.MAX_CONTEXT_TOKENS
      );

      const sources = [...new Set(usedResults.map(r => r.payload.document_name))];
      console.log(`📚 Sources: ${sources.join(', ')}`);
      console.log(`📏 Context tokens: ${tokenCount}`);

      // 3. Build efficient prompt - SINGLE CALL
      const systemPrompt = `You are an expert document analyst. Answer the user's question using ONLY the provided document excerpts.

INSTRUCTIONS:
- Provide a comprehensive, well-organized answer
- Use information from the excerpts below
- Organize into logical sections if needed
- Include relevant quotes when helpful
- Be thorough but concise
- Escape quotes properly for JSON

Respond in valid JSON format:
{
  "answer": "Complete answer in markdown format",
  "sources": ["doc1.pdf", "doc2.pdf"],
  "relevantSections": [
    {"documentName": "doc1.pdf", "section": "brief relevant excerpt"}
  ],
  "confidence": "high|medium|low",
  "searchStrategy": "brief description"
}`;

      const userPrompt = `QUESTION: ${query}

DOCUMENT EXCERPTS:
${context}

${historyContext ? `\nCONVERSATION CONTEXT:\n${historyContext}` : ''}`;

      const messages = [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: userPrompt }
      ];

      // 4. SINGLE LLM CALL
      const response = await this.llmCall(messages);
      
      // 5. Parse and return with token usage
      const parsedResponse = UtilService.parseLLMJson(response);
      
      const tokenUsage = {
        context: TokenEstimator.estimateTokens(systemPrompt + userPrompt),
        response: TokenEstimator.estimateTokens(response),
        total: TokenEstimator.estimateTokens(systemPrompt + userPrompt + response)
      };

      console.log(`📊 Token usage: ${tokenUsage.total} total (${tokenUsage.context} context + ${tokenUsage.response} response)`);

      return {
        answer: parsedResponse?.answer || "I encountered an error processing your question.",
        sources: parsedResponse?.sources || sources,
        relevantSections: parsedResponse?.relevantSections || [],
        searchStrategy: parsedResponse?.searchStrategy || `Analyzed ${usedResults.length} excerpts from ${sources.length} documents`,
        confidence: parsedResponse?.confidence || 'medium',
        tokenUsage: tokenUsage
      };

    } catch (error) {
      console.error('❌ Optimized AI response generation failed:', error);
      throw error;
    }
  }

  // STREAMING VERSION - Also single call
  static async generateResponseStreaming(query, historyContext = '', onChunk, onComplete) {
    try {
      console.log('🔄 Starting optimized streaming response...');
      
      // Same search and context building logic
      const searchResults = await VectorService.searchSimilar(query, this.CONFIG.SEARCH_LIMIT);
      
      const filteredResults = searchResults.filter(
        result => result.score >= this.CONFIG.MIN_SCORE_THRESHOLD
      );

      if (filteredResults.length === 0) {
        const noResultsMessage = "I couldn't find relevant information in your documents to answer this question.";
        
        // Simulate streaming
        const words = noResultsMessage.split(' ');
        for (const word of words) {
          onChunk(word + ' ');
          await new Promise(resolve => setTimeout(resolve, 50));
        }
        
        onComplete({
          answer: noResultsMessage,
          sources: [],
          relevantSections: [],
          searchStrategy: 'No relevant documents found',
          confidence: "low",
          tokenUsage: { context: 0, response: 0, total: 0 }
        });
        return;
      }

      const { context, usedResults, tokenCount } = TokenEstimator.buildContextWithTokenLimit(
        filteredResults, 
        this.CONFIG.MAX_CONTEXT_TOKENS
      );

      const sources = [...new Set(usedResults.map(r => r.payload.document_name))];
      
      // Build streaming prompt
      const systemPrompt = `
       You are a knowledgeable academic assistant. Your job is to write a comprehensive, well-organized, detailed answer to the question below, using ONLY the provided summaries.

  INSTRUCTIONS:
  - Write in clear, flowing prose (not JSON)
  - Combine all the relevant points
  - Write clearly and thoroughly
  - Organize into logical sections or paragraphs
  - Include important quotes if present
  - Ensure the answer is complete
  - Don't include document names in answer

  Write your response directly as text, not as JSON.`;

      const userPrompt = `QUESTION: ${query}

DOCUMENT EXCERPTS:
${context}

${historyContext ? `\nCONVERSATION CONTEXT:\n${historyContext}` : ''}`;

      const messages = [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: userPrompt }
      ];

      let fullAnswer = '';
      let responseTokens = 0;
      
      // Stream the response
      await this.streamingLlmCall(messages, (chunk) => {
        fullAnswer += chunk;
        responseTokens += TokenEstimator.estimateTokens(chunk);
        onChunk(chunk);
      });

      const tokenUsage = {
        context: TokenEstimator.estimateTokens(systemPrompt + userPrompt),
        response: responseTokens,
        total: TokenEstimator.estimateTokens(systemPrompt + userPrompt) + responseTokens
      };

      console.log(`📊 Streaming token usage: ${tokenUsage.total} total`);

      // Complete with metadata
      onComplete({
        answer: fullAnswer,
        sources: sources,
        relevantSections: this.extractRelevantSections(usedResults),
        searchStrategy: `Analyzed ${usedResults.length} excerpts from ${sources.length} documents`,
        confidence: this.assessConfidence(filteredResults, query),
        tokenUsage: tokenUsage
      });

    } catch (error) {
      console.error('❌ Streaming response generation failed:', error);
      onChunk('\n\n[Error occurred during streaming]');
      onComplete({
        answer: 'An error occurred while generating the response.',
        sources: [],
        relevantSections: [],
        searchStrategy: 'Error during processing',
        confidence: 'low',
        tokenUsage: { context: 0, response: 0, total: 0 }
      });
    }
  }

  // Streaming-only version of generateResponseWithSmartBatching
  static async generateResponseWithSmartBatching(query, historyContext = '', onChunk, onComplete) {
    try {
      console.log('🔄 Starting smart batched streaming response...');
      
      const searchResults = await VectorService.searchSimilar(query, this.CONFIG.SEARCH_LIMIT);
      
      if (searchResults.length === 0) {
        const noResultsMessage = "I couldn't find relevant information in your documents to answer this question.";
        
        // Simulate streaming for no results
        const words = noResultsMessage.split(' ');
        for (const word of words) {
          onChunk(word + ' ');
          await new Promise(resolve => setTimeout(resolve, 50));
        }
        
        onComplete({
          answer: noResultsMessage,
          sources: [],
          relevantSections: [],
          searchStrategy: 'No relevant documents found',
          confidence: "low",
          tokenUsage: { context: 0, response: 0, total: 0 }
        });
        return;
      }

      // Smart batching: only if we have too much content
      const totalContextTokens = TokenEstimator.estimateTokens(
        searchResults.map(r => r.payload.text).join('\n')
      );

      let processedContent;
      let batchTokens = 0;

      if (totalContextTokens > this.CONFIG.MAX_CONTEXT_TOKENS) {
        // Content too large - use batching
        console.log(`📊 Content too large (${totalContextTokens} tokens), using smart batching`);
        
        const BATCH_SIZE = 5;
        const batches = [];
        
        for (let i = 0; i < searchResults.length; i += BATCH_SIZE) {
          batches.push(searchResults.slice(i, i + BATCH_SIZE));
        }
        
        // Limit to 3 batches max to control cost
        const limitedBatches = batches.slice(0, 3);
        
        const summaries = await Promise.all(limitedBatches.map(
          batch => this.summarizeBatch(batch, query)
        ));

        processedContent = summaries.join('\n\n');
        batchTokens = summaries.length * 200; // Estimate batch call tokens
        
        console.log(`📊 Created ${summaries.length} summaries from ${limitedBatches.length} batches`);
      } else {
        // Use direct content - no batching needed
        processedContent = searchResults.map((result, index) => 
          `[${index + 1}] From "${result.payload.document_name}":\n${result.payload.text}`
        ).join('\n\n');
        
        console.log(`📊 Using direct content (${totalContextTokens} tokens)`);
      }

      // Prepare metadata
      const sources = [...new Set(searchResults.map(r => r.payload.document_name))];
      
      // Single, clean system prompt for streaming
      const systemPrompt = `You are an expert document analyst. Answer the user's question using the provided content.

  INSTRUCTIONS:
  - Write a comprehensive, well-organized answer
  - Use clear, flowing prose
  - Organize into logical sections or paragraphs  
  - Include relevant quotes when helpful
  - Be thorough but concise
  - Write directly as text, not as JSON

  Write your response as natural text that directly answers the question.`;

      const userPrompt = `QUESTION: ${query}

  CONTENT:
  ${processedContent}

  ${historyContext ? `\nCONVERSATION CONTEXT:\n${historyContext}` : ''}`;

      const messages = [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: userPrompt }
      ];

      // Stream the response
      let fullAnswer = '';
      let responseTokens = 0;
      
      await this.streamingLlmCall(messages, (chunk) => {
        fullAnswer += chunk;
        responseTokens += TokenEstimator.estimateTokens(chunk);
        onChunk(chunk);
      });

      // Calculate token usage
      const tokenUsage = {
        context: TokenEstimator.estimateTokens(systemPrompt + userPrompt),
        response: responseTokens,
        batching: batchTokens,
        total: TokenEstimator.estimateTokens(systemPrompt + userPrompt) + responseTokens + batchTokens
      };

      console.log(`📊 Streaming completed. Token usage: ${tokenUsage.total} total (${tokenUsage.batching} from batching)`);

      // Send completion metadata
      onComplete({
        answer: fullAnswer,
        sources: sources,
        relevantSections: this.extractRelevantSections(searchResults),
        searchStrategy: batchTokens > 0 
          ? `Smart batching: ${searchResults.length} excerpts processed with ${Math.floor(batchTokens/200)} summaries`
          : `Direct processing: ${searchResults.length} excerpts analyzed`,
        confidence: this.assessConfidence(searchResults, query),
        tokenUsage: tokenUsage
      });

    } catch (error) {
      console.error('❌ Smart batched streaming failed:', error);
      
      onChunk('\n\n[Error occurred during response generation]');
      onComplete({
        answer: 'An error occurred while generating the response.',
        sources: [],
        relevantSections: [],
        searchStrategy: 'Error during processing',
        confidence: 'low',
        tokenUsage: { context: 0, response: 0, total: 0 }
      });
    }
  }

  // Helper methods
  static extractRelevantSections(searchResults) {
    return searchResults.slice(0, 2).map(result => ({
      documentName: result.payload.document_name,
      section: result.payload.text.substring(0, 100) + "..."
    }));
  }

  static assessConfidence(searchResults) {
    if (searchResults.length === 0) return 'low';
    
    const avgScore = searchResults.reduce((sum, r) => sum + r.score, 0) / searchResults.length;
    const highQualityCount = searchResults.filter(r => r.score > 0.6).length;
    const sourceCount = new Set(searchResults.map(r => r.payload.document_name)).size;
    
    if (avgScore > 0.7 && highQualityCount >= 5 && sourceCount >= 2) return 'high';
    if (avgScore > 0.5 || (highQualityCount >= 3 && sourceCount >= 2)) return 'medium';
    return 'low';
  }
}

class SimpleTypeDetector {
  static detectType(filename, text) {
    console.log(`🔍 Detecting type for: ${filename}`);
    
    // Extract volume number from filename
    const volumeMatch = filename.match(/Volume[_\s]*(\d+)/i);
    const volumeNum = volumeMatch ? parseInt(volumeMatch[1]) : 0;
    
    // Simple volume-based detection
    if (volumeNum >= 13) {
      console.log(`📜 Volume ${volumeNum} = Parliamentary proceedings`);
      return 'parliamentary';
    }
    
    if (volumeNum === 11) {
      console.log(`📚 Volume ${volumeNum} = Book (Buddha and His Dhamma)`);
      return 'book';
    }
    
    if (volumeNum === 12) {
      console.log(`🎤 Volume ${volumeNum} = Speeches`);
      return 'speech';
    }
    
    // Quick content check for other volumes
    const firstPart = text.substring(0, 1000).toLowerCase();
    
    if (firstPart.includes('dr. ambedkar:') || firstPart.includes('chairman') || firstPart.includes('assembly')) {
      console.log(`📜 Content analysis = Parliamentary`);
      return 'parliamentary';
    }
    
    if (firstPart.includes('chapter') || firstPart.includes('preface') || firstPart.includes('table of contents')) {
      console.log(`📚 Content analysis = Book`);
      return 'book';
    }
    
    if (firstPart.includes('delivered at') || firstPart.includes('ladies and gentlemen')) {
      console.log(`🎤 Content analysis = Speech`);
      return 'speech';
    }
    
    console.log(`📝 Default = Essay/Article`);
    return 'essay';
  }
}

// Super Simple Chunking Service
class SimpleChunker {
  // Just 4 simple configs
  static CONFIGS = {
    parliamentary: { size: 512, overlap: 80 },   // Small for speaker turns
    book: { size: 1000, overlap: 200 },          // Large for complete thoughts
    speech: { size: 600, overlap: 120 },         // Medium for flow
    essay: { size: 800, overlap: 160 }           // Medium-large for arguments
  };

  static async chunkText(text, documentType) {
    const config = this.CONFIGS[documentType] || this.CONFIGS.essay;
    
    console.log(`⚙️ Chunking as ${documentType}: ${config.size} tokens, ${config.overlap} overlap`);
    
    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: config.size,
      chunkOverlap: config.overlap,
      separators: ['\n\n', '\n', '. ', '! ', '? ', '; ', ': ', ' ', '']
    });

    const chunks = await splitter.splitText(text);
    const validChunks = chunks.filter(chunk => chunk.trim().length > 10);
    
    console.log(`✂️ Created ${validChunks.length} chunks`);
    return validChunks;
  }
}

// Enhanced DocumentService (SIMPLE VERSION)
class EnhancedDocumentService {
  // Super simple processing
  static async processDocument(buffer, filename) {
    try {
      console.log(`📄 Processing: ${filename}`);
      
      // 1. Extract text
      const extraction = await DocumentService.extractTextFromPDF(buffer, filename);
      
      // 2. Detect type (super simple)
      const documentType = SimpleTypeDetector.detectType(filename, extraction.text);

      console.log(`📄 Detected type: ${documentType}`);
      
      // 3. Smart chunking
      const chunks = await SimpleChunker.chunkText(extraction.text, documentType);
      
      // 4. Build document (same format as before)
      const document = {
        id: randomUUID(),
        name: filename,
        status: 'processed',
        uploadedAt: new Date().toISOString(),
        documentType: documentType, // NEW: Document type
        metadata: {
          ...extraction.metadata,
          chunkCount: chunks.length,
          processingMethod: `Smart chunking (${documentType})`,
          chunkingStrategy: {
            type: documentType,
            chunkSize: SimpleChunker.CONFIGS[documentType]?.size || 800,
            overlap: SimpleChunker.CONFIGS[documentType]?.overlap || 160
          }
        },
        chunks
      };
      
      console.log(`✅ Processed: ${chunks.length} chunks (${documentType})`);
      return document;
      
    } catch (error) {
      console.error(`❌ Processing failed for ${filename}:`, error);
      throw error;
    }
  }
}

class AmbedkarPersonaService {
  
  // Enhanced search that maps modern topics to Ambedkar's relevant themes
  static TOPIC_MAPPINGS = {
    // Social Issues
    'social media': ['caste discrimination', 'untouchability', 'social reform', 'public opinion'],
    'online harassment': ['untouchability', 'social exclusion', 'dignity', 'human rights'],
    'cancel culture': ['social boycott', 'ostracism', 'freedom of speech'],
    'influencers': ['leadership', 'social reform', 'education', 'enlightenment'],
    
    // Political & Governance
    'democracy': ['parliamentary democracy', 'constitutional government', 'political participation'],
    'authoritarianism': ['dictatorship', 'fascism', 'individual liberty', 'constitutional safeguards'],
    'populism': ['majority rule', 'minority rights', 'constitutional morality'],
    'corruption': ['governance', 'public service', 'moral degradation'],
    
    // Technology & Society
    'artificial intelligence': ['education', 'knowledge', 'human dignity', 'employment'],
    'automation': ['labor', 'industrial development', 'unemployment', 'skill development'],
    'digital divide': ['education inequality', 'access to knowledge', 'social mobility'],
    
    // Economic Issues
    'inequality': ['economic justice', 'wealth distribution', 'labor rights'],
    'capitalism': ['industrial development', 'labor exploitation', 'economic reform'],
    'gig economy': ['labor conditions', 'worker rights', 'economic security'],
    
    // Social Justice
    'affirmative action': ['reservation', 'representation', 'compensatory discrimination'],
    'identity politics': ['caste politics', 'community representation', 'political participation'],
    'intersectionality': ['multiple identities', 'compound discrimination', 'women rights'],
    
    // Religion & Culture
    'secularism': ['religious freedom', 'separation of religion and state', 'minority rights'],
    'religious fundamentalism': ['religious fanaticism', 'communalism', 'tolerance'],
    'cultural appropriation': ['cultural identity', 'respect for traditions', 'dignity']
  };

  // Core themes that define Ambedkar's worldview
  static CORE_THEMES = [
    'constitutional democracy',
    'social justice',
    'human dignity',
    'equality',
    'education and enlightenment',
    'religious freedom',
    'labor rights',
    'women empowerment',
    'minority rights',
    'rule of law'
  ];

  // Ambedkar's characteristic phrases and concepts
  static SIGNATURE_CONCEPTS = [
    'constitutional morality',
    'educate, agitate, organize',
    'religion is for man, not man for religion',
    'political power is the master key',
    'liberty, equality, fraternity',
    'annihilation of caste',
    'separate electorate',
    'Buddhism as liberation'
  ];

  // Enhanced search strategy for persona responses
  static async searchForPersonaContext(query, originalSearchResults) {
    try {
      console.log('🔍 Searching for Ambedkar persona context...');
      
      // 1. Extract modern topics from query
      const modernTopics = this.extractModernTopics(query);
      console.log('🎯 Modern topics detected:', modernTopics);
      
      // 2. Map to Ambedkar's historical themes
      const historicalThemes = this.mapToHistoricalThemes(modernTopics);
      console.log('📚 Mapped historical themes:', historicalThemes);
      
      // 3. Search for additional context if needed
      let contextResults = [...originalSearchResults];
      
      // If we have historical themes, search for them too
      if (historicalThemes.length > 0) {
        for (const theme of historicalThemes.slice(0, 3)) { // Limit to avoid too many calls
          const themeResults = await VectorService.searchSimilar(theme, 5);
          contextResults.push(...themeResults);
        }
      }
      
      // 4. Remove duplicates and sort by relevance
      const uniqueResults = this.deduplicateResults(contextResults);
      
      // 5. Prioritize biographical and philosophical content
      const prioritizedResults = this.prioritizeContent(uniqueResults);
      
      console.log(`✅ Enhanced context: ${prioritizedResults.length} results`);
      return prioritizedResults;
      
    } catch (error) {
      console.error('❌ Persona context search failed:', error);
      return originalSearchResults; // Fallback to original results
    }
  }

  static extractModernTopics(query) {
    const topics = [];
    const lowerQuery = query.toLowerCase();
    
    for (const [modernTerm, themes] of Object.entries(this.TOPIC_MAPPINGS)) {
      if (lowerQuery.includes(modernTerm)) {
        topics.push(modernTerm);
      }
    }
    
    return topics;
  }

  static mapToHistoricalThemes(modernTopics) {
    const themes = new Set();
    
    modernTopics.forEach(topic => {
      const mappedThemes = this.TOPIC_MAPPINGS[topic] || [];
      mappedThemes.forEach(theme => themes.add(theme));
    });
    
    return Array.from(themes);
  }

  static deduplicateResults(results) {
    const seen = new Set();
    return results.filter(result => {
      const key = `${result.payload.document_name}-${result.payload.chunk_index}`;
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });
  }

  static prioritizeContent(results) {
    // Sort by relevance score first, then prioritize certain document types
    return results.sort((a, b) => {
      // Higher score wins
      if (b.score !== a.score) return b.score - a.score;
      
      // Prioritize speeches and parliamentary debates for persona
      const aIsSpeech = a.payload.document_name.toLowerCase().includes('speech') || 
                      a.payload.document_name.toLowerCase().includes('volume12');
      const bIsSpeech = b.payload.document_name.toLowerCase().includes('speech') || 
                      b.payload.document_name.toLowerCase().includes('volume12');
      
      if (aIsSpeech && !bIsSpeech) return -1;
      if (!aIsSpeech && bIsSpeech) return 1;
      
      return 0;
    });
  }

  // Generate the persona-specific system prompt
  static generatePersonaPrompt(query, contextualInfo = '') {
    const currentYear = new Date().getFullYear();
    
    return `You are Dr. Bhimrao Ramji Ambedkar, speaking in ${currentYear}. You have been given the context of modern times and are responding based on your lifelong philosophy, writings, and speeches.

CORE IDENTITY:
- You are the principal architect of the Indian Constitution
- Champion of social justice, equality, and human dignity
- Advocate for the oppressed and marginalized
- Believer in constitutional democracy and rule of law
- Convert to Buddhism who saw it as a path to liberation
- Scholar, lawyer, and social reformer

SPEAKING STYLE AND APPROACH:
- Give DIRECT answers first, then provide context and nuance
- Use first person ("I believe", "In my experience", "Yes, this happened to me")
- Reference your actual works and speeches when relevant
- Don't deflect or avoid difficult questions
- Be specific about your experiences when asked directly
- Then explain the broader principles and implications
- Use measured, scholarly language but with conviction

ANSWER STRUCTURE:
1. **Direct Answer**: Start with a clear, specific response to the question
2. **Personal Experience**: Share relevant personal experiences when applicable
3. **Broader Context**: Explain the larger social/political implications
4. **Modern Application**: Connect to current issues and your principles

RESPONSE EXAMPLES:
- If asked "Were you discriminated against?" → Start with "Yes, I faced severe discrimination throughout my life..."
- If asked "Do you support reservations?" → Start with "Yes, I believe reservations are necessary..."
- If asked about specific incidents → Give the specific answer first, then context

CORE PRINCIPLES TO EMBODY:
1. Constitutional morality and rule of law
2. Liberty, equality, and fraternity
3. Education as the tool for liberation
4. Political power as the master key to progress
5. Annihilation of caste and social hierarchy
6. Religious freedom and individual dignity
7. Labor rights and economic justice
8. Women's rights and empowerment

CRITICAL: Answer the specific question asked directly and honestly, then provide the deeper analysis. Avoid academic evasion.

${contextualInfo ? `ADDITIONAL CONTEXT: ${contextualInfo}` : ''}

Respond as Dr. Ambedkar would, giving clear answers while maintaining philosophical depth.`;
  }

  // Enhanced response generation for persona
  static async generatePersonaResponse(query, historyContext = '', onChunk, onComplete) {
    try {
      console.log('🎭 Generating Ambedkar persona response...');
      
      // 1. Initial search
      const searchResults = await VectorService.searchSimilar(query, AIService.CONFIG.SEARCH_LIMIT);
      
      if (searchResults.length === 0) {
        const noResultsMessage = "I apologize, but I cannot find sufficient information in my writings and speeches to address your question adequately. Perhaps you could ask about my documented work on constitutional law, social justice, or religious philosophy?";
        
        // Stream the response
        const words = noResultsMessage.split(' ');
        for (const word of words) {
          onChunk(word + ' ');
          await new Promise(resolve => setTimeout(resolve, 50));
        }
        
        onComplete({
          answer: noResultsMessage,
          sources: [],
          relevantSections: [],
          searchStrategy: 'Insufficient content in documented works',
          confidence: 'low',
          tokenUsage: { context: 0, response: 0, batching: 0, total: 0 },
          personaType: 'ambedkar'
        });
        return;
      }

      // 2. Enhanced context search for persona
      const enhancedResults = await this.searchForPersonaContext(query, searchResults);
      
      // 3. Filter by score threshold
      const filteredResults = enhancedResults.filter(
        result => result.score >= AIService.CONFIG.MIN_SCORE_THRESHOLD
      );

      // 4. Smart batching logic (similar to original)
      const totalContextTokens = TokenEstimator.estimateTokens(
        filteredResults.map(r => r.payload.text).join('\n')
      );

      let processedContent;
      let batchTokens = 0;

      if (totalContextTokens > AIService.CONFIG.MAX_CONTEXT_TOKENS) {
        console.log(`📊 Using smart batching for persona context`);
        
        const BATCH_SIZE = 4; // Smaller batches for persona to maintain voice
        const batches = [];
        
        for (let i = 0; i < filteredResults.length; i += BATCH_SIZE) {
          batches.push(filteredResults.slice(i, i + BATCH_SIZE));
        }
        
        const limitedBatches = batches.slice(0, 3);
        
        // Custom summarization for persona
        const summaries = await Promise.all(limitedBatches.map(
          batch => this.summarizeForPersona(batch, query)
        ));

        processedContent = summaries.join('\n\n');
        batchTokens = summaries.length * 200;
        
      } else {
        // Direct content for persona
        processedContent = filteredResults.map((result, index) => 
          `[Excerpt ${index + 1}] From "${result.payload.document_name}":\n${result.payload.text}`
        ).join('\n\n');
      }

      // 5. Generate persona-specific prompt
      const sources = [...new Set(filteredResults.map(r => r.payload.document_name))];
      const modernContext = this.generateModernContextNote(query);
      
      const systemPrompt = this.generatePersonaPrompt(query, modernContext);
      
      const userPrompt = `QUESTION: ${query}

MY WRITINGS AND SPEECHES (for reference):
${processedContent}

${historyContext ? `\nPREVIOUS CONVERSATION:\n${historyContext}` : ''}

Please respond as Dr. Ambedkar, applying your documented philosophy to this modern question.`;

      const messages = [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: userPrompt }
      ];

      // 6. Stream the persona response
      let fullAnswer = '';
      let responseTokens = 0;
      
      await AIService.streamingLlmCall(messages, (chunk) => {
        fullAnswer += chunk;
        responseTokens += TokenEstimator.estimateTokens(chunk);
        onChunk(chunk);
      });

      // 7. Complete with metadata
      const tokenUsage = {
        context: TokenEstimator.estimateTokens(systemPrompt + userPrompt),
        response: responseTokens,
        batching: batchTokens,
        total: TokenEstimator.estimateTokens(systemPrompt + userPrompt) + responseTokens + batchTokens
      };

      onComplete({
        answer: fullAnswer,
        sources: sources,
        relevantSections: this.extractRelevantSections(filteredResults),
        searchStrategy: `Persona response based on ${filteredResults.length} excerpts from Ambedkar's works`,
        confidence: this.assessPersonaConfidence(filteredResults, query),
        tokenUsage: tokenUsage,
        personaType: 'ambedkar'
      });

    } catch (error) {
      console.error('❌ Persona response generation failed:', error);
      
      onChunk('\n\n[I apologize, but I encountered an error while accessing my writings.]');
      onComplete({
        answer: 'I apologize, but I encountered an error while formulating my response.',
        sources: [],
        relevantSections: [],
        searchStrategy: 'Error during persona response generation',
        confidence: 'low',
        tokenUsage: { context: 0, response: 0, batching: 0, total: 0 },
        personaType: 'ambedkar'
      });
    }
  }

  // Custom summarization that preserves Ambedkar's voice
  static async summarizeForPersona(batchChunks, query) {
    const batchText = batchChunks
      .map((r, i) => `[${i+1}] ${r.payload.text}`)
      .join("\n\n");

    const systemPrompt = `You are preparing excerpts from Dr. Ambedkar's writings for him to reference when answering: "${query}"

FOCUS ON:
- Specific incidents, experiences, or events Ambedkar mentions
- His direct statements and positions on issues
- Concrete examples and personal experiences
- Clear policy positions and recommendations
- His exact quotes when possible

PRESERVE:
- Factual details and specific incidents
- His direct statements and clear positions
- Personal experiences and anecdotes
- Specific examples and evidence
- His reasoning and logical arguments

Extract information that helps him give direct, specific answers rather than just general principles.`;
    
    const userPrompt = `EXCERPTS FROM AMBEDKAR'S WRITINGS:\n${batchText}`;
    
    const messages = [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: userPrompt }
    ];
    
    return await AIService.llmCall(messages);
  }

  static generateModernContextNote(query) {
    const lowerQuery = query.toLowerCase();
    let contextNotes = [];
    
    // Add context about modern technologies/concepts
    if (lowerQuery.includes('social media') || lowerQuery.includes('internet')) {
      contextNotes.push('Modern digital platforms for communication and information sharing');
    }
    
    if (lowerQuery.includes('ai') || lowerQuery.includes('artificial intelligence')) {
      contextNotes.push('Automated systems that can perform tasks requiring human intelligence');
    }
    
    if (lowerQuery.includes('globalization')) {
      contextNotes.push('Worldwide economic and cultural integration');
    }
    
    return contextNotes.join('. ');
  }

  static extractRelevantSections(searchResults) {
    return searchResults.slice(0, 3).map(result => ({
      documentName: result.payload.document_name,
      section: result.payload.text.substring(0, 150) + "..."
    }));
  }

  static assessPersonaConfidence(searchResults, query) {
    if (searchResults.length === 0) return 'low';
    
    // Check if we have results from multiple types of documents
    const docTypes = new Set();
    searchResults.forEach(result => {
      const docName = result.payload.document_name.toLowerCase();
      if (docName.includes('speech') || docName.includes('volume12')) docTypes.add('speeches');
      if (docName.includes('volume11')) docTypes.add('books');
      if (docName.includes('parliament') || docName.match(/volume1[3-9]/)) docTypes.add('parliamentary');
      else docTypes.add('writings');
    });
    
    const avgScore = searchResults.reduce((sum, r) => sum + r.score, 0) / searchResults.length;
    const highQualityCount = searchResults.filter(r => r.score > 0.6).length;
    
    // High confidence: Good scores + multiple document types + sufficient results
    if (avgScore > 0.65 && highQualityCount >= 4 && docTypes.size >= 2) return 'high';
    
    // Medium confidence: Decent coverage
    if (avgScore > 0.5 && highQualityCount >= 2) return 'medium';
    
    return 'low';
  }
}

// API Routes

// Health check
app.get('/api/health', (req, res) => {
  res.json({ 
    status: 'ok',
    services: {
      qdrant: qdrantClient !== null,
      embeddings: embeddingModel !== null
    },
    timestamp: new Date().toISOString()
  });
});

// Initialize services
app.post('/api/initialize', async (req, res) => {
  try {
    await initializeServices();
    res.json({ success: true, message: 'Services initialized' });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
});

app.get('/api/config', async (req, res) => {
  try {
    await new Promise((resolve) => setTimeout(resolve(), 300))
    res.json({ 
      success: true, 
      config: {
        isUploadEnabled: process.env.UPLOAD_ENABLED === 'true'
      }
    });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
});

app.get('/api/ping', (req, res) => {
  res.json({ success: true, status: 'alive', timestamp: Date.now() });
});

// Upload document
app.post('/api/documents/upload', upload.single('file'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No file uploaded' });
    }
    
    // Process document completely on backend
    const document = await EnhancedDocumentService.processDocument(req.file.buffer, req.file.originalname);
    
    // Save to vector store
    await VectorService.saveDocument(document);
    
    // Return minimal response - no chunks exposed
    res.json({
      success: true,
      document: {
        id: document.id,
        name: document.name,
        uploadedAt: document.uploadedAt,
        status: 'ready'
      }
    });
  } catch (error) {
    console.error('Document upload failed:', error);
    res.status(500).json({ 
      success: false, 
      error: error.message 
    });
  }
});

// Get document list from cache
app.get('/api/documents', async (req, res) => {
  try {
    const cachedData = DocumentService.getDocumentsFromCache();
    res.json({ documents: cachedData.documents || [] });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Get latest document list 
app.post('/api/documents/refresh', async (req, res) => {
  try {
    const documents = await VectorService.refreshDocumentCache();
    res.json({ 
      success: true, 
      message: 'Document cache refreshed',
      count: documents.length,
      lastUpdated: cacheLastUpdated
    });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
});

// Delete document
app.delete('/api/documents/:id', async (req, res) => {
  try {
    await VectorService.deleteDocument(req.params.id);
    res.json({ success: true });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
});

// Ambedkar Persona Streaming Chat Endpoint
app.post('/api/chat/ambedkar', async (req, res) => {
  try {
    const { query, historyContext = '' } = req.body;
    
    if (!query) {
      return res.status(400).json({ error: 'Query is required' });
    }

    console.log('🎭 Starting Ambedkar persona chat for query:', query.substring(0, 100) + '...');

    // Set headers for Server-Sent Events
    res.setHeader('Content-Type', 'text/plain; charset=utf-8');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Headers', 'Cache-Control');

    let hasStarted = false;

    // Handle streaming chunks
    const onChunk = (chunk) => {
      if (!hasStarted) {
        hasStarted = true;
        console.log('🎭 Ambedkar persona streaming started...');
      }
      
      const data = {
        type: 'chunk',
        content: chunk,
        persona: 'ambedkar'
      };
      
      res.write(`data: ${JSON.stringify(data)}\n\n`);
    };

    // Handle completion
    const onComplete = (metadata) => {
      console.log('✅ Ambedkar persona response completed');
      
      const completionData = {
        type: 'complete',
        sources: metadata.sources || [],
        confidence: metadata.confidence || 'medium',
        relevantSections: metadata.relevantSections || [],
        searchStrategy: metadata.searchStrategy || 'Ambedkar persona response',
        tokenUsage: metadata.tokenUsage || { context: 0, response: 0, batching: 0, total: 0 },
        personaType: 'ambedkar'
      };
      
      res.write(`data: ${JSON.stringify(completionData)}\n\n`);
      res.write('data: [DONE]\n\n');
      res.end();
    };

    // Generate persona response
    await AmbedkarPersonaService.generatePersonaResponse(query, historyContext, onChunk, onComplete);

  } catch (error) {
    console.error('💥 Ambedkar persona chat failed:', error);
    
    const errorData = {
      type: 'error',
      error: error.message,
      persona: 'ambedkar'
    };
    
    res.write(`data: ${JSON.stringify(errorData)}\n\n`);
    res.write('data: [DONE]\n\n');
    res.end();
  }
});

// NEW: Streaming chat endpoint
app.post('/api/chat/stream', async (req, res) => {
  try {
    const { query, historyContext = '' } = req.body;
    
    if (!query) {
      return res.status(400).json({ error: 'Query is required' });
    }

    console.log('🔄 Starting streaming chat for query:', query.substring(0, 100) + '...');

    // Set headers for Server-Sent Events
    res.setHeader('Content-Type', 'text/plain; charset=utf-8');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Headers', 'Cache-Control');

    let hasStarted = false;
    let streamedContent = '';

    // Handle streaming chunks
    const onChunk = (chunk) => {
      if (!hasStarted) {
        hasStarted = true;
        console.log('📡 Starting to stream response...');
      }
      
      streamedContent += chunk; // Keep track of streamed content
      
      const data = {
        type: 'chunk',
        content: chunk
      };
      
      res.write(`data: ${JSON.stringify(data)}\n\n`);
    };

    // Handle completion
    const onComplete = (metadata) => {
      console.log('✅ Streaming completed, sending metadata...');
      console.log('📦 Final metadata:', metadata);
      
      // Send completion data with all metadata
      const completionData = {
        type: 'complete',
        sources: metadata.sources || [],
        confidence: metadata.confidence || 'medium',
        relevantSections: metadata.relevantSections || [],
        searchStrategy: metadata.searchStrategy || `Analyzed streaming response for: ${query.substring(0, 50)}...`,
        tokenUsage: metadata.tokenUsage || { context: 0, response: 0, total: 0 }
      };
      
      console.log('📤 Sending completion data:', completionData);
      res.write(`data: ${JSON.stringify(completionData)}\n\n`);
      
      // Send done signal
      res.write('data: [DONE]\n\n');
      res.end();
    };

    // Start streaming
    await AIService.generateResponseWithSmartBatching(query, historyContext, onChunk, onComplete);

  } catch (error) {
    console.error('💥 Streaming chat failed:', error);
    
    const errorData = {
      type: 'error',
      error: error.message
    };
    
    res.write(`data: ${JSON.stringify(errorData)}\n\n`);
    res.write('data: [DONE]\n\n');
    res.end();
  }
});

// Regular chat endpoint (fallback)
app.post('/api/chat', async (req, res) => {
  try {
    const { query, historyContext = '' } = req.body;
    
    if (!query) {
      return res.status(400).json({ error: 'Query is required' });
    }
    
    const response = await AIService.generateResponse(query, historyContext);
    res.json(response);
  } catch (error) {
    console.error('Chat failed:', error);
    res.status(500).json({ error: error.message });
  }
});

// Error handling
app.use((err, req, res, next) => {
  console.error('Unhandled error:', err);
  res.status(500).json({ error: 'Internal server error' });
});

// Start server
const startServer = async () => {
  try {
    console.log('🚀 Starting server...');
    
    // Enable manual garbage collection
    if (global.gc) {
      console.log('♿️ Garbage collection enabled');
      setInterval(() => {
        global.gc();
        const memUsage = process.memoryUsage();
        console.log(`📏 Memory: RSS ${Math.round(memUsage.rss / 1024 / 1024)}MB, Heap ${Math.round(memUsage.heapUsed / 1024 / 1024)}MB`);
      }, 60000); // Log memory every minute
    }

    await initializeServices();
    
    app.listen(PORT, '0.0.0.0', () => {
      console.log(`✅ Server running on port ${PORT}`);
      console.log(`📡 Streaming endpoint available at /api/chat/stream`);
      console.log(`💾 Memory optimized for 512MB limit`);
    });
  } catch (error) {
    console.error('💥 Failed to start server:', error);
    process.exit(1);
  }
};

startServer();
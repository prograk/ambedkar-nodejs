/* eslint-disable no-undef */
import express from 'express';
import cors from 'cors';
import multer from 'multer';
import { QdrantClient } from '@qdrant/js-client-rest';
import { pipeline, env } from '@xenova/transformers';
import { config } from 'dotenv';
import { randomUUID } from 'crypto';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';

config();

const app = express();
const PORT = process.env.PORT || 3001;

// Configure multer for file uploads
const upload = multer({ 
  storage: multer.memoryStorage(),
  limits: { fileSize: 50 * 1024 * 1024 }, // 50MB limit
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
  chunkSize: 1000,
  chunkOverlap: 200,
  maxChunks: 100 // Prevent abuse
};

// Global instances
let qdrantClient = null;
let embeddingModel = null;
let isModelLoading = false;

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
    
    // Initialize embedding model
    await initializeEmbeddingModel();
    
    return true;
  } catch (error) {
    console.error('❌ Service initialization failed:', error);
    throw error;
  }
};

const initializeEmbeddingModel = async () => {
  if (embeddingModel) return embeddingModel;
  
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
    
    embeddingModel = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', {
      dtype: 'fp32',
      device: 'cpu'
    });
    
    console.log('✅ Embedding model loaded');
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
      // const data = await pdf(buffer);
      
      if (!data.text || data.text.trim().length === 0) {
        throw new Error('PDF contains no extractable text');
      }
      
      return {
        text: data.text,
        pages: data.numpages,
        metadata: {
          filename,
          extractedAt: new Date().toISOString(),
          characterCount: data.text.length
        }
      };
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
        separators: ['\n\n', '\n', '.', '!', '?', ';', ':', ' ', ''],
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
}

class EmbeddingService {
  static async generateEmbeddings(texts) {
    try {
      const model = await initializeEmbeddingModel();
      const embeddings = [];
      
      // Process in small batches
      const batchSize = 5;
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
      
      // Upload in batches
      const batchSize = 100;
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
        limit: limit,
        with_payload: true,
        score_threshold: 0.2
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
        limit: 1000,
        with_payload: true,
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

class AIService {
  static async generateResponse(query, historyContext = '') {
    try {
      // Search for relevant content
      const searchResults = await VectorService.searchSimilar(query, 15);
      
      if (searchResults.length === 0) {
        return {
          answer: "I couldn't find relevant information in your documents to answer this question.",
          sources: [],
          relevantSections: '',
          searchStrategy: '',
          confidence: "low"
        };
      }
      
      // Build context from search results
      const context = searchResults.map((result, index) => 
        `[${index + 1}] From "${result.payload.document_name}":\n${result.payload.text}`
      ).join('\n\n');
      
      const sources = [...new Set(searchResults.map(r => r.payload.document_name))];
      
      // Build prompt
      const systemPrompt = `You are an expert document analyst with access to relevant document excerpts.

INSTRUCTIONS:
1. Analyze document excerpts for information relevant to the user's question
2. Provide comprehensive answers citing source documents
3. If information spans multiple documents, synthesize and compare
4. Be thorough but concise
5. Always cite sources by document name
6. Don't return response content enclosed inside \`\`\`json\`\`\` 

Respond with JSON in this exact format:
{
  "answer": "Your detailed answer here",
  "sources": ["document1.pdf", "document2.pdf"],
  "relevantSections": [
    {
      "document": "document1.pdf", 
      "section": "Brief excerpt of relevant text"
    }
  ],
  "confidence": "high|medium|low",
  "searchStrategy": "Brief description of how you found the information"
}

CRITICAL: Output ONLY valid JSON. No other text or formatting.`;


const userPrompt = `DOCUMENT EXCERPTS:
${context}

${historyContext}

USER QUESTION: ${query}

Please analyze the excerpts and respond with the JSON format specified.`;
      
      // Call OpenRouter API
      const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${process.env.OPENROUTER_API_KEY}`,
          'Content-Type': 'application/json',
          'HTTP-Referer': process.env.FRONTEND_URL || 'http://localhost:3000',
          'X-Title': 'Document Chat API',
        },
        body: JSON.stringify({
          model: process.env.OPENROUTER_MODEL,
          messages: [
            { role: 'system', content: systemPrompt },
            { role: 'user', content: userPrompt }
          ],
          max_tokens: 1500,
          temperature: 0.1
        })
      });

      if (!response.ok) {
        throw new Error(`OpenRouter API error: ${response.status}`);
      }

      const data = await response.json();
      const responseText = data?.choices[0]?.message?.content || '';
      
      try {
        const parsedResponse = JSON.parse(responseText);
        return {
          answer: parsedResponse.answer,
          sources: parsedResponse.sources || sources,
          relevantSections: parsedResponse.relevantSections || [],
          searchStrategy: parsedResponse.searchStrategy || '',
          confidence: parsedResponse.confidence || 'medium'
        };
      // eslint-disable-next-line no-unused-vars
      } catch (parseError) {
        // Fallback if JSON parsing fails
        return {
          answer: responseText || "I encountered an error processing your question.",
          sources: sources,
          relevantSections: [],
          searchStrategy: `Analyzed ${searchResults.length} excerpts`,
          confidence: "low"
        };
      }
    } catch (error) {
      console.error('AI response generation failed:', error);
      throw error;
    }
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
    res.json({ success: true, config: {
      isUploadEnabled: false
    }});
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
});

// Upload document
app.post('/api/documents/upload', upload.single('file'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No file uploaded' });
    }
    
    // Process document completely on backend
    const document = await DocumentService.processDocument(req.file.buffer, req.file.originalname);
    
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

// Get document list  
app.get('/api/documents', async (req, res) => {
  try {
    const documents = await VectorService.getDocumentList();
    res.json({ documents });
  } catch (error) {
    res.status(500).json({ error: error.message });
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

// Chat endpoint
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
    
    await initializeServices();
    
    app.listen(PORT, '0.0.0.0', () => {
      console.log(`✅ Server running on port ${PORT}`);
    });
  } catch (error) {
    console.error('💥 Failed to start server:', error);
    process.exit(1);
  }
};

startServer();
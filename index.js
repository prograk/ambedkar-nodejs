/* eslint-disable no-undef */
import express from 'express';
import cors from 'cors';
import multer from 'multer';
import { QdrantClient } from '@qdrant/js-client-rest';
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
  maxChunks: 100, // Prevent abuse
  separators: ["\n\n", "\n", "।", ".", "?", "!"],
};

// Global instances
let qdrantClient = null;
let embeddingModel = null;
let isModelLoading = false;
let documentCache = [];
let cacheLastUpdated = null;

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
    
    // Initialize embedding model
    // await initializeEmbeddingModel();
    
    await VectorService.refreshDocumentCache();

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
    
    embeddingModel = await pipeline('feature-extraction', 'Xenova/paraphrase-multilingual-MiniLM-L12-v2', {
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
      // return new Promise((resolve, reject) => {
      //   const pdfParser = new PDFParser();
        
      //   pdfParser.on('pdfParser_dataError', (errData) => {
      //     reject(new Error(`PDF parsing error: ${errData.parserError}`));
      //   });
        
      //   pdfParser.on('pdfParser_dataReady', (pdfData) => {
      //     try {
      //       // Extract text from parsed PDF data
      //       let text = '';
            
      //       if (pdfData.Pages && pdfData.Pages.length > 0) {
      //         pdfData.Pages.forEach(page => {
      //           if (page.Texts && page.Texts.length > 0) {
      //             page.Texts.forEach(textItem => {
      //               if (textItem.R && textItem.R.length > 0) {
      //                 textItem.R.forEach(r => {
      //                   if (r.T) {
      //                     text += decodeURIComponent(r.T) + ' ';
      //                   }
      //                 });
      //               }
      //             });
      //             text += '\n';
      //           }
      //         });
      //       }
            
      //       if (!text || text.trim().length === 0) {
      //         reject(new Error('PDF contains no extractable text'));
      //         return;
      //       }
            
      //       resolve({
      //         text: text.trim(),
      //         pages: pdfData.Pages.length,
      //         metadata: {
      //           filename,
      //           extractedAt: new Date().toISOString(),
      //           characterCount: text.length
      //         }
      //       });
      //     } catch (parseError) {
      //       reject(new Error(`Failed to process PDF data: ${parseError.message}`));
      //     }
      //   });
        
      //   // Parse the PDF buffer
      //   pdfParser.parseBuffer(buffer);
      // });
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
      documentCache = await this.getDocumentList();
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

class AIService {
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

  static async generateResponse(query, historyContext = '') {
    try {
      // Search for relevant content
      const searchResults = await VectorService.searchSimilar(query, 30);

      if (searchResults.length === 0) {
        return {
          answer: "I couldn't find relevant information in your documents to answer this question.",
          sources: [],
          relevantSections: '',
          searchStrategy: '',
          confidence: "low"
        };
      }

      const BATCH_SIZE = 5;
      const batches = [];

      for (let i = 0; i < searchResults.length; i += BATCH_SIZE) {
        const batch = searchResults.slice(i, i + BATCH_SIZE);
        batches.push(batch);
      }

      const summaries = await Promise.all(batches.map(
        batch => this.summarizeBatch(batch, query)
      ));  

      const combinedSummaries = summaries.join("\n\n");

      console.log(combinedSummaries, "combinedSummaries")
      
      // Build context from search results
      const context = searchResults.map((result, index) => 
        `[${index + 1}] From "${result.payload.document_name}":\n${result.payload.text}`
      ).join('\n\n');
      
      const sources = [...new Set(searchResults.map(r => r.payload.document_name))];
      
      // Build prompt
      const systemPrompt = `
You are a knowledgeable academic assistant. Your job is to write a comprehensive, well-organized, detailed answer to the question below, using ONLY the provided summaries.

INSTRUCTIONS:
- Combine all the relevant points
- Write clearly and thoroughly
- Organize into logical sections or paragraphs
- Include important quotes if present
- Ensure the answer is complete so the user doesn't need to read the documents.
- Add escape in quote so that JSON can be parsed
- releventSections max 2 allowed in output
- Don't include document names in answer

Respond with JSON in this exact format:
{
  "answer": "string (MUST be a complete, detailed answer in Markdown format, bullet points, numbered lists, quotes if relevant)",
  "sources": ["volume1.pdf", "volume2.pdf"],
  "relevantSections": [
    { 
      "documentName": "volume1.pdf", 
      "section": "Brief 20 words excerpt of relevant text just for refernce purpose, no markdown"
    }
  ],
  "confidence": "high|medium|low",
  "searchStrategy": "Brief description of how you found the information"
}`;

// HISTORY CONTEXT: ${historyContext}

const userPrompt = `
QUESTION: ${query}

DOCUMENT CONTEXT: ${context}

COLLECTED SUMMARIES: ${combinedSummaries}
`;

      const messages = [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: userPrompt }
      ]
      
      // Call OpenRouter API
      const response = await this.llmCall(messages);
      
      try {
        const parsedResponse = UtilService.parseLLMJson(response);

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
          answer: "I encountered an error processing your question.",
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
        id: crypto.randomUUID(),
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
        isUploadEnabled: false
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
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import List, Dict, Any, AsyncGenerator, Generator
import json
from app.core.config import settings


class LLMService:
    def __init__(self):
        self.llm = ChatOllama(
            model=settings.LLM_MODEL, base_url=settings.OLLAMA_BASE_URL, temperature=0.3
        )

    def generate_response(self, prompt: str, system_prompt: str = "") -> str:
        """Generate response from LLM"""
        messages = []

        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))

        messages.append(HumanMessage(content=prompt))

        response = self.llm.invoke(messages)
        if hasattr(response, "content"):
            return str(response.content)
        return str(response)

    def stream_response(
        self, prompt: str, system_prompt: str = ""
    ) -> Generator[str, None, None]:
        """Stream response from LLM"""
        messages = []

        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))

        messages.append(HumanMessage(content=prompt))

        for chunk in self.llm.stream(messages):
            if hasattr(chunk, "content") and chunk.content:
                content = str(chunk.content)
                yield content

    async def astream_response(
        self, prompt: str, system_prompt: str = ""
    ) -> AsyncGenerator[str, None]:
        """Async stream response from LLM"""
        messages = []

        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))

        messages.append(HumanMessage(content=prompt))

        async for chunk in self.llm.astream(messages):
            if hasattr(chunk, "content") and chunk.content:
                content = str(chunk.content)
                yield content

    def create_document_analysis_prompt(
        self, document_content: str, analysis_type: str = "summary"
    ) -> str:
        """Create prompt for document analysis"""
        if analysis_type == "summary":
            return f"""
            Please analyze the following document and provide a comprehensive summary:
            
            Document Content:
            {document_content}
            
            Please provide:
            1. A brief overview (2-3 sentences)
            2. Key points and findings
            3. Main topics covered
            4. Any notable patterns or insights
            
            Format your response as a structured JSON with the following keys:
            - overview: Brief summary
            - key_points: List of main points
            - topics: List of topics covered
            - insights: List of notable insights
            """

        elif analysis_type == "data_analysis":
            return f"""
            Analyze the following data and provide insights:
            
            Data:
            {document_content}
            
            Please provide:
            1. Data type and structure analysis
            2. Statistical summary (if applicable)
            3. Patterns and trends
            4. Anomalies or outliers
            5. Recommendations for visualization
            
            Format your response as structured JSON.
            """

        else:
            return f"Analyze the following content: {document_content}"

    def create_rag_prompt(
        self, context: str, question: str, chat_history: List[Dict] | None = None
    ) -> str:
        """Create RAG prompt with context and question"""
        history_text = ""
        if chat_history:
            history_text = "\n".join(
                [
                    f"{msg['role']}: {msg['content']}"
                    for msg in chat_history[-5:]  # Last 5 messages
                ]
            )

        return f"""
        You are an AI assistant specialized in document analysis and question answering.
        Use the following context to answer the user's question. If the context doesn't 
        contain enough information to answer the question, say so politely.
        
        Context:
        {context}
        
        Previous Conversation:
        {history_text}
        
        User Question: {question}
        
        Provide a helpful and accurate answer based on the context. Include specific 
        details from the document when relevant. If you reference specific parts of 
        the document, mention that you're doing so.
        """

    def create_visualization_recommendation_prompt(
        self, data_description: str, data_sample: str
    ) -> str:
        """Create prompt for visualization recommendations"""
        return f"""
        Based on the following data information, recommend the most appropriate 
        visualizations and analysis approaches:
        
        Data Description:
        {data_description}
        
        Sample Data:
        {data_sample}
        
        Please recommend:
        1. Top 3 most suitable chart types for this data
        2. Key metrics and statistics to calculate
        3. Interesting patterns or relationships to explore
        4. Data preprocessing steps if needed
        
        Format your response as JSON with these sections.
        """

    def extract_json_from_response(self, response: str) -> Dict[str, Any]:
        """Extract JSON from LLM response"""
        try:
            # Try to find JSON in the response
            start_idx = response.find("{")
            end_idx = response.rfind("}") + 1

            if start_idx != -1 and end_idx != -1:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
            else:
                # If no JSON found, return raw response
                return {"raw_response": response}

        except json.JSONDecodeError:
            # If JSON parsing fails, return raw response
            return {"raw_response": response}

    def get_token_count(self, text: str) -> int:
        """Estimate token count (rough approximation)"""
        # Simple approximation: ~4 characters per token
        return len(text) // 4

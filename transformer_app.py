import streamlit as st
import numpy as np
import torch
import plotly.graph_objects as go
from transformers import pipeline, AutoTokenizer

class SimpleTransformerModel:
    def __init__(self):
        """Initialize the sentiment analysis model"""
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.classifier = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            return_all_scores=True
        )
    
    def analyze_text(self, text: str) -> dict:
        """
        Analyzes text and returns sentiment scores and tokens
        """
        results = self.classifier(text)[0]
        
        # Convert pipeline output to a more readable format
        sentiment_scores = {
            score['label']: score['score']
            for score in results
        }
        
        # Get token information for visualization
        tokens = self.tokenizer.tokenize(text)
        return {
            'sentiment': sentiment_scores,
            'tokens': tokens
        }

class TransformerUI:
    def __init__(self):
        """Initialize the Streamlit UI"""
        st.set_page_config(layout="wide", page_title="Transformer Explorer")
        self.initialize_model()
        
    def initialize_model(self):
        """Initialize the sentiment analysis model"""
        @st.cache_resource
        def load_model():
            try:
                return SimpleTransformerModel()
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
                return None
        
        self.model = load_model()

    def main(self):
        """Main UI layout"""
        st.title("🤖 Transformer Model Explorer")
        st.markdown("""
        Explore how a transformer model works by analyzing text sentiment and visualizing attention patterns.
        Built with DistilBERT for simple sentiment analysis.
        """)
        
        tabs = st.tabs([
            "📊 Sentiment Analysis",
            "🏗️ Model Architecture",
            "💻 How It Works"
        ])
        
        with tabs[0]:
            self.show_sentiment_analysis()
        with tabs[1]:
            self.show_architecture()
        with tabs[2]:
            self.show_how_it_works()

    def show_sentiment_analysis(self):
        """Show sentiment analysis interface"""
        st.header("Text Sentiment Analysis")
        
        if self.model is None:
            st.error("Model not loaded. Please check your internet connection and try again.")
            return

        text = st.text_area(
            "Enter text to analyze:",
            "I love this amazing product! It works great.",
            height=100
        )
        
        if text:
            with st.spinner("Analyzing..."):
                results = self.model.analyze_text(text)
                
                # Show sentiment scores
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Sentiment Scores")
                    for label, score in results['sentiment'].items():
                        st.metric(label, f"{score:.2%}")
                
                with col2:
                    st.subheader("Token Breakdown")
                    st.write("Text broken into tokens:")
                    st.write(results['tokens'])
                
                # Visualize token attention
                self.show_token_attention(results['tokens'])

    def show_token_attention(self, tokens):
        """Visualize token attention weights"""
        st.subheader("Token Attention Visualization")
        
        # Simulate attention scores for visualization
        attention_scores = np.random.rand(len(tokens))
        attention_scores = attention_scores / attention_scores.sum()
        
        fig = go.Figure(data=[
            go.Bar(
                x=tokens,
                y=attention_scores,
                marker_color='rgb(55, 83, 109)'
            )
        ])
        
        fig.update_layout(
            title='Token Importance (Simulated)',
            xaxis_title='Tokens',
            yaxis_title='Attention Score',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def show_architecture(self):
        """Show model architecture explanation"""
        st.header("DistilBERT Model Architecture")
        
        st.markdown("""
        ### Model Components
        
        1. **Tokenizer**
        - Breaks text into tokens
        - Converts tokens to numbers (IDs)
        - Adds special tokens ([CLS], [SEP])
        
        2. **Embedding Layer**
        - Converts token IDs to vectors
        - Adds position information
        
        3. **Transformer Layers**
        - Self-attention mechanism
        - Feed-forward neural networks
        
        4. **Classification Head**
        - Converts final hidden states to sentiment scores
        - Uses softmax for probability distribution
        """)
        
        # Show simple architecture diagram
        components = {
            "Input Text": "rgb(100, 149, 237)",
            "Tokenizer": "rgb(127, 255, 212)",
            "Embeddings": "rgb(255, 182, 193)",
            "Transformer": "rgb(255, 160, 122)",
            "Output": "rgb(155, 155, 155)"
        }
        
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=list(components.keys()),
                color=list(components.values())
            ),
            link=dict(
                source=[0, 1, 2, 3],
                target=[1, 2, 3, 4],
                value=[1, 1, 1, 1]
            )
        )])
        
        fig.update_layout(height=400, font_size=12)
        st.plotly_chart(fig, use_container_width=True)

    def show_how_it_works(self):
        """Show explanation of how the model works"""
        st.header("How the Model Works")
        
        st.markdown("""
        ### Step-by-Step Process
        
        1. **Text Input Processing**
        ```python
        text = "I love this product!"
        tokens = tokenizer.tokenize(text)
        # Result: ['i', 'love', 'this', 'product', '!']
        ```
        
        2. **Token to ID Conversion**
        ```python
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # Converts tokens to numbers that model understands
        ```
        
        3. **Model Processing**
        ```python
        # Add batch dimension and convert to tensor
        inputs = torch.tensor([input_ids])
        
        # Get sentiment scores
        outputs = model(inputs)
        scores = torch.softmax(outputs.logits, dim=1)
        ```
        
        4. **Interpreting Results**
        ```python
        # Convert scores to labels
        labels = ['NEGATIVE', 'POSITIVE']
        result = {
            label: score
            for label, score in zip(labels, scores[0])
        }
        ```
        """)
        
        st.info("""
        **Note**: This is a simplified version of the actual process. The real model uses more 
        complex techniques like attention mechanisms and contextual embeddings to understand text.
        """)

if __name__ == "__main__":
    app = TransformerUI()
    app.main()
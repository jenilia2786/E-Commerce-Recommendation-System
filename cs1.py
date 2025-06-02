import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import pandas as pd
import random

# 1. Define 100+ training examples with intents
intents_data = {
    'greeting': ['Hi', 'Hello', 'Hey', 'Hi there', 'Good morning', 'Good afternoon', 'Howdy',
                 'Greetings', 'Yo', 'Hey there', 'Hi bot', 'Hello bot', 'Hey assistant', 'Hey buddy'],
    'product_info': ['Tell me about the product', 'What is this item?', 'Can I get details?', 'Product specifications',
                     'Give me product info', 'Explain the product', 'What are the features?', 'Is this product good?',
                     'What does this product do?', 'Tell me more about it', 'Describe this product', 'How is this item?',
                     'Item description please', 'What is included?'],
    'order_status': ['Where is my order?', 'Track my order', 'Order tracking', 'Status of my purchase',
                     'Is my order shipped?', 'When will I receive my order?', 'Order update', 'Has it been delivered?',
                     'Give order status', 'I want to know my order status', 'Check delivery status', 'Shipping status please',
                     'Is it out for delivery?', 'Order progress'],
    'return_policy': ['What is the return policy?', 'How do I return an item?', 'Return instructions',
                      'Can I return this product?', 'Is return allowed?', 'Return this order', 'Return process',
                      'How to get a refund?', 'Can I get my money back?', 'What is your refund policy?',
                      'I want to return something', 'Return eligibility?', 'Help with return', 'Return window'],
    'troubleshoot': ['Product not working', 'Issue with item', 'Something is wrong', 'I need troubleshooting help',
                     'The item is broken', 'Facing a problem', 'It is not functioning', 'Problem with my order',
                     'I have a complaint', 'Support for technical issue', 'Product defective', 'Help me fix it',
                     'There is an error', 'How do I fix it?'],
    'escalation': ['Talk to a human', 'Escalate issue', 'I need real person', 'Let me speak to support',
                   'I want human help', 'Escalate to agent', 'Speak to representative', 'Connect me to customer care',
                   'I need more help', 'I need to file a complaint', 'This is not helping', 'Customer service please',
                   'Let me escalate this', 'Raise a ticket'],
    'thanks': ['Thanks', 'Thank you', 'Appreciate it', 'Thanks a lot', 'Thank you so much',
               'Much obliged', 'Great help', 'Thanks buddy', 'Cheers', 'Thanks for helping',
               'I am grateful', 'Many thanks', 'Nice work', 'You helped me a lot'],
    'goodbye': ['Bye', 'Goodbye', 'See you later', 'Talk to you later', 'Catch you later',
                'I am done', 'Thanks, bye', 'Have a good day', 'That’s all', 'See you soon',
                'Farewell', 'Take care', 'Later', 'I am logging off']
}

# 2. Create dataset
patterns, labels = [], []
for intent, examples in intents_data.items():
    for example in examples:
        patterns.append(example.lower())
        labels.append(intent)

df = pd.DataFrame({'pattern': patterns, 'intent': labels})

# 3. Train classifier
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(max_iter=1000))
])
pipeline.fit(df['pattern'], df['intent'])

# 4. Responses for each intent
responses = {
    'greeting': ['Hello! 👋 How can I assist you today?', 'Hi there! Need help with something?'],
    'product_info': ['This product has high-quality features you’ll love!', 'Here are the specifications: durable, lightweight, and affordable.'],
    'order_status': ['Please provide your order ID and I’ll check it for you.', 'Checking your order... may I have the order number?'],
    'return_policy': ['We offer a 30-day return window. Items must be in original condition.', 'You can start a return via your account > Orders.'],
    'troubleshoot': ['Sorry to hear that. Try restarting the device. Need more help?', 'Can you describe the issue in detail? I’ll do my best to help.'],
    'escalation': ['Connecting you to a support agent now... 🔄', 'Escalating your request. Please hold on.'],
    'thanks': ['You’re welcome! 😊', 'Glad to help!', 'Anytime!'],
    'goodbye': ['Goodbye! Take care 👋', 'See you next time!', 'Happy to help!']
}

# 5. Streamlit conversational UI
st.set_page_config(page_title="Conversational Support Bot", layout="centered")
st.title("🤖 Customer Support Chatbot")
st.markdown("Ask me anything related to your orders, returns, products, or support!")

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Chat input
user_input = st.chat_input("Type your message here...")

if user_input:
    intent = pipeline.predict([user_input.lower()])[0]
    bot_reply = random.choice(responses[intent])

    # Add to chat history
    st.session_state.chat_history.append(("user", user_input))
    st.session_state.chat_history.append(("bot", bot_reply))

# Display chat history
for role, message in st.session_state.chat_history:
    if role == "user":
        st.chat_message("user").markdown(message)
    else:
        st.chat_message("assistant").markdown(message)

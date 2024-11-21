import sqlite3
import tweepy
import pandas as pd
from datetime import datetime, timedelta
from collections import Counter
import re
from sklearn.ensemble import RandomForestClassifier
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from telegram import Bot
from telethon import TelegramClient
import smtplib
from email.mime.text import MIMEText
import asyncio

# ----------------------------
# 1. CONFIGURACIÓN DE BASE DE DATOS
# ----------------------------

# Conexión SQLite
conn = sqlite3.connect('crypto_trends.db')
cursor = conn.cursor()

# Crear tabla para palabras clave
cursor.execute('''
    CREATE TABLE IF NOT EXISTS keywords (
        id INTEGER PRIMARY KEY,
        keyword TEXT UNIQUE,
        last_checked TIMESTAMP,
        frequency INTEGER,
        sentiment REAL
    )
''')
conn.commit()

# ----------------------------
# 2. CONEXIÓN A APIs
# ----------------------------

API_KEY = 'your_api_key'
API_SECRET = 'your_api_secret'
ACCESS_TOKEN = 'your_access_token'
ACCESS_TOKEN_SECRET = 'your_access_token_secret'

# Autenticación de Twitter con Tweepy
auth = tweepy.OAuthHandler(API_KEY, API_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)

def fetch_tweets(keyword, max_tweets=100):
    """Captura tweets desde Twitter API"""
    tweets_data = []
    for tweet in tweepy.Cursor(api.search_tweets, q=keyword, lang='en', tweet_mode='extended').items(max_tweets):
        tweets_data.append(tweet.full_text)
    return tweets_data


# Configuración de la API de Telegram
api_id = 123456
api_hash = "abcdef1234567890abcdef1234567890"

async def fetch_messages(chat, limit=10):
    async with TelegramClient("my_session", api_id, api_hash) as client:
        messages = await client.get_messages(chat, limit=limit)
        return [msg.message for msg in messages if msg.message]


# ----------------------------
# 3. ANÁLISIS DE SENTIMIENTO
# ----------------------------

analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(texts):
    """Calcula el sentimiento promedio usando VADER."""
    scores = [analyzer.polarity_scores(text)['compound'] for text in texts]
    return sum(scores) / len(scores) if scores else 0

# ----------------------------
# 4. FILTRO DE PALABRAS CLAVE
# ----------------------------

def extract_keywords(texts, existing_keywords):
    """Extrae palabras clave nuevas."""
    all_words = ' '.join(texts).lower()
    words = re.findall(r'\b[a-zA-Z]+\b', all_words)
    common_words = Counter(words).most_common(20)
    new_keywords = [word for word, count in common_words if word not in existing_keywords]
    return new_keywords

# ----------------------------
# 5. ENTRENAMIENTO DEL MODELO
# ----------------------------

# Datos históricos simulados
training_data = pd.DataFrame({
    'frequency': [100, 200, 1500, 800, 50],
    'sentiment': [0.8, 0.6, 0.7, 0.9, 0.4],
    'success': [1, 1, 1, 1, 0]
})

# Entrenamiento del modelo
X_train = training_data[['frequency', 'sentiment']]
y_train = training_data['success']
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ----------------------------
# 6. GUARDADO Y ANÁLISIS DE PALABRAS CLAVE
# ----------------------------

def save_keyword(keyword, frequency, sentiment):
    """Guarda o actualiza palabras clave en la base de datos."""
    now = datetime.now()
    cursor.execute('''
        INSERT INTO keywords (keyword, last_checked, frequency, sentiment)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(keyword) DO UPDATE SET 
            last_checked=excluded.last_checked,
            frequency=excluded.frequency,
            sentiment=excluded.sentiment
    ''', (keyword, now, frequency, sentiment))
    conn.commit()

def analyze_keyword(keyword, max_tweets=100):
    """Analiza una palabra clave con frecuencia, sentimiento y predicción."""
    tweets = fetch_tweets(keyword, max_tweets)
    frequency = len(tweets)
    sentiment = analyze_sentiment(tweets)
    prediction = model.predict([[frequency, sentiment]])[0]
    save_keyword(keyword, frequency, sentiment)
    return {
        'keyword': keyword,
        'frequency': frequency,
        'sentiment': sentiment,
        'prediction': 'Exitoso' if prediction == 1 else 'No exitoso'
    }

# ----------------------------
# 7. NOTIFICACIONES
# ----------------------------
# Función asincrónica que envía el mensaje
async def send_telegram_update(chat_id, message):
    """Envía un mensaje a través de Telegram utilizando async."""
    token = 'tu_bot_token_aqui'  # Reemplaza con tu token de bot
    bot = Bot(token=token)
    
    try:
        # Usamos 'await' para llamar a la función asincrónica
        await bot.send_message(chat_id=chat_id, text=message)
        print("Mensaje enviado con éxito.")
    except Exception as e:
        print(f"Error al enviar el mensaje: {e}")




def send_email_notification(subject, message, recipient_email):
    """Envía una notificación por correo electrónico."""
    sender_email = "your_email@gmail.com"
    sender_password = "your_email_password"

    msg = MIMEText(message)
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = recipient_email

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, recipient_email, msg.as_string())

# ----------------------------
# 8. PIPELINE COMPLETO
# ----------------------------

def process_pipeline(max_tweets=100):
    """Pipeline completo de detección y análisis."""
    # Obtener palabras clave existentes
    cursor.execute('SELECT keyword FROM keywords')
    existing_keywords = [row[0] for row in cursor.fetchall()]

    # Captar textos
    tweets = fetch_tweets('crypto', max_tweets)
    new_keywords = extract_keywords(tweets, existing_keywords)

    # Analizar palabras clave nuevas
    for keyword in new_keywords:
        analysis = analyze_keyword(keyword)
        print(f"Análisis para {keyword}: {analysis}")

        # Notificar si es relevante
        if analysis['frequency'] > 100 and analysis['sentiment'] > 0.8:
            
            # Función principal asincrónica
            async def main():
                chat_id = '123456789'  # Reemplaza con el chat_id correcto
                message = f"La palabra '{keyword}' muestra alta relevancia.\nSentimiento: {analysis['sentiment']}\nFrecuencia: {analysis['frequency']}"
                await send_telegram_update(chat_id, message)
            asyncio.run(main())

            send_email_notification(
                subject=f"Tendencia detectada: {keyword}",
                message=f"La palabra '{keyword}' muestra alta relevancia.\nSentimiento: {analysis['sentiment']}\nFrecuencia: {analysis['frequency']}",
                recipient_email='recipient@example.com'
            )

# Ejecutar pipeline
process_pipeline()
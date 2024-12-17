from flask import Flask, render_template, request, jsonify, redirect, url_for
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pickle
import re
from datetime import datetime
import os
import joblib
import csv


app = Flask(__name__)

MODEL_PATH = 'models/message_classifier.pkl'
VECTORIZER_PATH = 'models/tfidf_vectorizer.pkl'
ENCODER_PATH = 'models/label_encoder.pkl'
DATASET_PATH = 'FINAL-SMS-DATA.csv'

class MessageClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        
  
        self.patterns = {
            'OTP': [
                r'\b[Oo][Tt][Pp]\b.*?(\d{4,6})',
                r'\b(verification code|code|pin)\b.*?(\d{4,6})',
                r'(\d{4,6}).*?\b(is your|is the|as your)[^.]*?(code|OTP|pin|password)\b',
                r'\b(one time password)\b.*?(\d{4,6})'
            ],
            'LOGISTICS': [
                r'\b(order|delivery|shipment|package|tracking)\b.*?(#\w+|\b[A-Z]{2}\d{9,12}[A-Z]{2}\b)',
                r'\b(delivered|shipped|dispatched|out for delivery)\b',
                r'\b(courier|delivery partner)\b'
            ],
            'BILL': [

                r'\b(bill|payment|due|amount)\b.*?(Rs\.?\s*\d+|\$\s*\d+|INR\s*\d+)',
                r'\b(pay|clear|settle)\b.*?(bill|payment|due)',
                r'\b(electricity|gas|water|phone)\b.*?(bill|payment)',
                r'(Rs\.?\s*\d+|\$\s*\d+|INR\s*\d+).*?\b(due|payable|outstanding)\b',           
                r'\b(recharge|top[- ]?up)\b.*?(mobile|phone|dth|data)',
                r'\b(book|booking)\b.*?(ticket|movie|show|flight|train|bus)',
                r'\b(transfer|sent|received)\b.*?(Rs\.?\s*\d+|\$\s*\d+|INR\s*\d+)',
                r'\b(transaction|payment)\b.*?(success|confirmed|completed)',
                r'\b(wallet|upi|bank)\b.*?(transfer|payment|transaction)',
                r'\b(subscription|renewal|plan)\b.*?(due|expiring|activated)',
                r'\b(insurance|premium|emi)\b.*?(due|payment|installment)',
                r'\b(credit|debit)\b.*?(card|payment|transaction)',
                r'\b(fee|tuition|admission)\b.*?(payment|due|deposit)',
                r'\b(rent|maintenance)\b.*?(payment|due|deposit)'
            ],
            'PROMOTIONAL': [
                r'\b(off|discount|sale|offer|cashback)\b',
                r'\b(limited time|special offer|exclusive)\b',
                r'%\s*off',
                r'\b(free|complimentary)\b'
            ]
        }

    
        self.spam_patterns = {
            'SPAM': [
                r'\b(lottery|winner|won|prize|claim)\b',
                r'\b(click|link|subscribe|unsubscribe)\b',
                r'\b(investment|invest|stocks|forex)\b',
                r'\b(casino|gambling|bet)\b',
                r'\b(loan|credit|debt|mortgage)\b',
                r'\b(sex|dating|hot|singles)\b',
                r'\b(weight loss|diet|slim)\b',
                r'\b(earn money|make money|income)\b',
                r'\b(urgent|action required|immediate)\b',
                r'\b(congratulations|congrats)\b.*?(won|selected|chosen)',
                r'^\s*(?:re:|fwd:)',
                r'\b(viagra|cialis|pharmacy)\b',
                r'\b(work from home|job offer)\b',
                r'\b(limited time offer|act now|hurry)\b'
            ]
        }

    def extract_actionable_items(self, text, category):
        """
        Extract actionable items from the message based on category
        Returns a dictionary of possible actions
        """
        actions = {}
        
        if category == 'BILL':
          
            if re.search(r'\b(bill|payment|due)\b', text, re.IGNORECASE):
                actions['pay_bill'] = True
                
       
            recharge_match = re.search(r'\b(recharge|top[- ]?up)\b.*?(mobile|phone|dth|data)', text, re.IGNORECASE)
            if recharge_match:
                actions['recharge'] = True
                actions['recharge_type'] = recharge_match.group(2)
                
          
            booking_match = re.search(r'\b(book|booking)\b.*?(ticket|movie|show|flight|train|bus)', text, re.IGNORECASE)
            if booking_match:
                actions['book'] = True
                actions['booking_type'] = booking_match.group(2)
                
        
            if re.search(r'\b(transfer|sent|received)\b', text, re.IGNORECASE):
                actions['view_transaction'] = True
                
          
            if re.search(r'\b(subscription|renewal|plan)\b', text, re.IGNORECASE):
                actions['manage_subscription'] = True
                
          
            if re.search(r'\b(insurance|premium|emi)\b', text, re.IGNORECASE):
                actions['view_insurance'] = True
                
          
            amount_match = re.search(r'(Rs\.?\s*\d+|\$\s*\d+|INR\s*\d+)', text)
            if amount_match:
                actions['amount'] = amount_match.group(1)
                
        return actions

    def is_spam(self, text, category):
       
        for pattern in self.spam_patterns['SPAM']:
            if re.search(pattern, text, re.IGNORECASE):
                return True
                
       
        if category == 'OTP':
            if len(re.findall(r'\d{4,6}', text)) > 1:
                return True
            if re.search(r'\b(urgent|winner|claim|prize)\b', text, re.IGNORECASE):
                return True
        
        elif category == 'LOGISTICS':
            if re.search(r'\b(claim|prize|winner|urgent)\b.*?(delivery|package)', text, re.IGNORECASE):
                return True
        
        elif category == 'BILL':
            if re.search(r'\b(urgent|immediate|warning|last chance)\b.*?(payment|bill)', text, re.IGNORECASE):
                return True
            if re.search(r'\b(account.*?suspended|service.*?terminated)\b', text, re.IGNORECASE):
                return True
        
        elif category == 'PROMOTIONAL':
            if re.search(r'\b(free|win|congratulations|selected|chosen)\b.*?(prize|money|gift)', text, re.IGNORECASE):
                return True
            if re.search(r'\b(limited.*?offer|only.*?left|last.*?chance)\b', text, re.IGNORECASE):
                return True
            
        return False

    def train(self, X, y):
        X_processed = [self.preprocess_text(text) for text in X]
        X_tfidf = self.vectorizer.fit_transform(X_processed)
        y_encoded = self.label_encoder.fit_transform(y)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_tfidf, y_encoded, test_size=0.2, random_state=42
        )
        
        self.classifier.fit(X_train, y_train)
        
        y_pred = self.classifier.predict(X_test)
        metrics = {
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        if not os.path.exists('models'):
            os.makedirs('models')
            
        joblib.dump(self.classifier, MODEL_PATH)
        joblib.dump(self.vectorizer, VECTORIZER_PATH)
        joblib.dump(self.label_encoder, ENCODER_PATH)
        
        self.is_trained = True
        return metrics

    def preprocess_text(self, text):
        if isinstance(text, str):
            text = text.lower()
            text = ' '.join(text.split())
            return text
        return str(text)

    def pattern_match(self, text):
        text = str(text)
        for category, patterns in self.patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return category
        return None

    def predict(self, text):
        if not self.is_trained:
            raise Exception("Model not trained yet!")
            
        pattern_category = self.pattern_match(text)
        if pattern_category:
            is_spam = self.is_spam(text, pattern_category)
            actionable_items = self.extract_actionable_items(text, pattern_category)
            return pattern_category, 0.95, is_spam, actionable_items
            
        text_processed = self.preprocess_text(text)
        X_tfidf = self.vectorizer.transform([text_processed])
        prediction = self.classifier.predict(X_tfidf)
        predicted_label = self.label_encoder.inverse_transform(prediction)[0]
        proba = float(np.max(self.classifier.predict_proba(X_tfidf)))
        is_spam = self.is_spam(text, predicted_label)
        actionable_items = self.extract_actionable_items(text, predicted_label)
        
        return str(predicted_label), proba, is_spam, actionable_items

class MessageManager:
    def __init__(self):
        self.messages = []
        
    def add_message(self, text, category, confidence, is_spam, highlighted_text=None, otp=None, 
                   tracking_number=None, bill_amount=None, promo_code=None, actionable_items=None):
        message = {
            'id': len(self.messages) + 1,
            'text': text,
            'category': category,
            'confidence': confidence,
            'is_spam': is_spam,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'processed': False,
            'highlighted_text': highlighted_text,
            'otp': otp,
            'tracking_number': tracking_number,
            'bill_amount': bill_amount,
            'promo_code': promo_code,
            'actionable_items': actionable_items
        }
        self.messages.append(message)
        return message
    
    def get_messages(self):
        """
        Returns all messages in reverse chronological order
        """
        return list(reversed(self.messages))

    def clear_messages(self):
        """
        Clears all messages
        """
        self.messages = []


message_classifier = MessageClassifier()
message_manager = MessageManager()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        message_text = request.form.get('message')
        if message_text:
            # Classify the message
            category, confidence, is_spam, actionable_items = message_classifier.predict(message_text)
            
            # Extract additional information based on category
            otp = None
            tracking_number = None
            bill_amount = None
            promo_code = None
            highlighted_text = None
            
            if category == 'OTP':
                otp_match = re.search(r'\b\d{4,6}\b', message_text)
                if otp_match:
                    otp = otp_match.group()
                    highlighted_text = message_text.replace(otp, f'<mark>{otp}')
            
            elif category == 'LOGISTICS':
                tracking_match = re.search(r'\b[A-Z0-9]{8,14}\b', message_text)
                if tracking_match:
                    tracking_number = tracking_match.group()
                    highlighted_text = message_text.replace(tracking_number, f'<mark>{tracking_number}')
            
            elif category == 'BILL':
                amount_match = re.search(r'(Rs\.?\s*\d+|\$\s*\d+|INR\s*\d+)', message_text)
                if amount_match:
                    bill_amount = amount_match.group()
                    highlighted_text = message_text.replace(bill_amount, f'<mark>{bill_amount}')
            
            elif category == 'PROMOTIONAL':
                promo_match = re.search(r'\b[A-Z0-9]{4,10}\b', message_text)
                if promo_match:
                    promo_code = promo_match.group()
                    highlighted_text = message_text.replace(promo_code, f'<mark>{promo_code}')
            
           
            message_manager.add_message(
                text=message_text,
                category=category,
                confidence=confidence,
                is_spam=is_spam,
                highlighted_text=highlighted_text,
                otp=otp,
                tracking_number=tracking_number,
                bill_amount=bill_amount,
                promo_code=promo_code,
                actionable_items=actionable_items
            )
            
            return redirect(url_for('index'))
    
  
    messages = message_manager.get_messages()
    return render_template('index.html', messages=messages)

@app.route('/clear', methods=['POST'])
def clear_messages():
    message_manager.clear_messages()
    return redirect(url_for('index'))


print("Loading dataset and training model...")
try:
    df = pd.read_csv(DATASET_PATH)
    print("Available columns:", df.columns.tolist())
    print("Sample data:")
    print(df.head())
    
    def categorize_message(text):
        text = str(text)
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in message_classifier.patterns['OTP']):
            return 'OTP'
        elif any(re.search(pattern, text, re.IGNORECASE) for pattern in message_classifier.patterns['LOGISTICS']):
            return 'LOGISTICS'
        elif any(re.search(pattern, text, re.IGNORECASE) for pattern in message_classifier.patterns['BILL']):
            return 'BILL'
        elif any(re.search(pattern, text, re.IGNORECASE) for pattern in message_classifier.patterns['PROMOTIONAL']):
            return 'PROMOTIONAL'
        else:
            return 'OTHER'

    messages = df['_body'].fillna('')
    categories = [categorize_message(text) for text in messages]
    
    metrics = message_classifier.train(messages, categories)
    print("Model trained successfully!")
    print("\nTraining Metrics:")
    print(metrics)
    
except Exception as e:
    print(f"Error loading dataset or training model: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True, port=8080)

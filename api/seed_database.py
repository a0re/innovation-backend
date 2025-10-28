#!/usr/bin/env python3
"""
Seed the database with fake prediction data for testing.
Usage: python seed_database.py
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from faker import Faker
import random
from datetime import datetime, timedelta
from database import init_database, save_prediction

fake = Faker()

# Spam message templates
SPAM_TEMPLATES = [
    "Congratulations! You have won ${amount}! Click here to claim your prize now!",
    "URGENT: Your account has been suspended. Verify your identity immediately at {url}",
    "FREE MONEY! Call {phone} now to get your cash prize. Limited time offer!",
    "You have been selected for a special offer. Reply with your bank details to {email}",
    "WIN A FREE {product}! Click this link now before it expires: {url}",
    "Act now! Limited time offer expires in {hours} hours. Call {phone} immediately.",
    "Your package is waiting. Click {url} to claim your delivery reward.",
    "Congratulations! You've been selected as a lucky winner. Text WIN to {phone}",
    "ALERT: Suspicious activity on your account. Verify now at {url} or account will be locked",
    "Get rich quick! Invest ${amount} today and earn ${amount2} tomorrow! {url}",
    "Claim your tax refund of ${amount} now! Click here: {url}",
    "You've inherited ${amount} from a distant relative. Contact {email} to claim.",
    "Special discount! Get {percent}% off {product}! Limited stock! Buy now: {url}",
    "Your subscription is expiring! Renew now to avoid service interruption: {url}",
    "You have {count} new messages waiting. View them here: {url}",
]

# Not spam message templates
NOT_SPAM_TEMPLATES = [
    "Hey, are you free for lunch tomorrow?",
    "The meeting has been rescheduled to {time}.",
    "Thanks for your help with the project!",
    "Can you send me the report when you get a chance?",
    "Happy birthday! Hope you have a great day!",
    "Reminder: Doctor's appointment tomorrow at {time}.",
    "Your order #{order_id} has been shipped and will arrive in 2-3 business days.",
    "Meeting notes from yesterday's standup are attached.",
    "Hey {name}, do you want to grab coffee this afternoon?",
    "The presentation went great, thanks for your support!",
    "Just wanted to check in and see how you're doing.",
    "Can we reschedule our call to {time}?",
    "Thanks for sending over those files!",
    "Looking forward to working with you on this project.",
    "Your flight is confirmed for {date}. Have a great trip!",
    "Dinner at {time} tonight? Let me know!",
    "The package you ordered will be delivered tomorrow.",
    "Great job on the presentation yesterday!",
]

def generate_spam_message():
    """Generate a fake spam message"""
    template = random.choice(SPAM_TEMPLATES)
    return template.format(
        amount=random.randint(100, 10000),
        amount2=random.randint(10000, 100000),
        url=f"http://{fake.domain_name()}/{fake.uri_path()}",
        phone=fake.phone_number(),
        email=fake.email(),
        product=random.choice(["IPHONE", "LAPTOP", "CAR", "VACATION", "GIFT CARD"]),
        hours=random.randint(1, 48),
        percent=random.choice([50, 60, 70, 80, 90]),
        count=random.randint(5, 99)
    )

def generate_not_spam_message():
    """Generate a fake legitimate message"""
    template = random.choice(NOT_SPAM_TEMPLATES)
    return template.format(
        time=fake.time(pattern="%I:%M %p"),
        name=fake.first_name(),
        order_id=fake.random_int(min=10000, max=99999),
        date=fake.date(pattern="%B %d")
    )

def generate_model_results(is_spam):
    """Generate fake model prediction results"""
    # Models are generally correct but not perfect
    base_confidence = random.uniform(0.7, 0.95) if is_spam else random.uniform(0.65, 0.9)

    # Add some noise to individual models
    nb_confidence = base_confidence + random.uniform(-0.1, 0.1)
    lr_confidence = base_confidence + random.uniform(-0.1, 0.1)
    svc_confidence = base_confidence + random.uniform(-0.1, 0.1)

    # Clamp confidences
    nb_confidence = max(0.5, min(0.99, nb_confidence))
    lr_confidence = max(0.5, min(0.99, lr_confidence))
    svc_confidence = max(0.5, min(0.99, svc_confidence))

    # Occasionally make models disagree
    if random.random() < 0.15:  # 15% disagreement
        nb_spam = not is_spam if random.random() < 0.5 else is_spam
        lr_spam = not is_spam if random.random() < 0.5 else is_spam
        svc_spam = not is_spam if random.random() < 0.5 else is_spam
    else:
        nb_spam = lr_spam = svc_spam = is_spam

    spam_votes = sum([nb_spam, lr_spam, svc_spam])

    return {
        'multinomial_nb': {
            'prediction': 'spam' if nb_spam else 'not spam',
            'confidence': nb_confidence,
            'is_spam': nb_spam
        },
        'logistic_regression': {
            'prediction': 'spam' if lr_spam else 'not spam',
            'confidence': lr_confidence,
            'is_spam': lr_spam
        },
        'linear_svc': {
            'prediction': 'spam' if svc_spam else 'not spam',
            'confidence': svc_confidence,
            'is_spam': svc_spam
        },
        'ensemble': {
            'prediction': 'spam' if spam_votes >= 2 else 'not spam',
            'confidence': (nb_confidence + lr_confidence + svc_confidence) / 3,
            'is_spam': spam_votes >= 2,
            'spam_votes': spam_votes,
            'total_votes': 3
        }
    }

def seed_database(num_records=500):
    """Seed the database with fake predictions"""
    print(f"🌱 Seeding database with {num_records} fake predictions...")

    # Initialize database
    init_database()
    print("✅ Database initialized")

    # Generate predictions with varying timestamps over the past 30 days
    spam_count = 0
    not_spam_count = 0

    for i in range(num_records):
        # 40% spam, 60% not spam (realistic distribution)
        is_spam = random.random() < 0.4

        if is_spam:
            message = generate_spam_message()
            spam_count += 1
        else:
            message = generate_not_spam_message()
            not_spam_count += 1

        # Generate model results
        model_results = generate_model_results(is_spam)
        ensemble = model_results['ensemble']

        # Generate timestamp (distributed over past 30 days)
        days_ago = random.randint(0, 30)
        hours_ago = random.randint(0, 23)
        minutes_ago = random.randint(0, 59)
        timestamp = datetime.now() - timedelta(days=days_ago, hours=hours_ago, minutes=minutes_ago)

        # Cluster ID for spam (0-4 clusters)
        cluster_id = random.randint(0, 4) if is_spam else None

        # Save to database
        save_prediction(
            message=message,
            processed_message=message.lower(),  # Simple preprocessing
            prediction=ensemble['prediction'],
            confidence=ensemble['confidence'],
            is_spam=ensemble['is_spam'],
            model_results=model_results,
            cluster_id=cluster_id
        )

        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{num_records} predictions...")

    print(f"\n✅ Successfully seeded database!")
    print(f"📊 Statistics:")
    print(f"   - Total: {num_records}")
    print(f"   - Spam: {spam_count} ({spam_count/num_records*100:.1f}%)")
    print(f"   - Not Spam: {not_spam_count} ({not_spam_count/num_records*100:.1f}%)")
    print(f"\n💡 View the data:")
    print(f"   - API: http://localhost:8000/stats")
    print(f"   - Dashboard: http://localhost:5174/dashboard")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Seed the spam detection database with fake data")
    parser.add_argument(
        "--count",
        type=int,
        default=500,
        help="Number of fake predictions to generate (default: 500)"
    )

    args = parser.parse_args()

    seed_database(args.count)

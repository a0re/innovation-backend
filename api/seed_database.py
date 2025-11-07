#!/usr/bin/env python3
"""
Seed the database with fake prediction data for testing.
Usage: python seed_database.py --count 500
"""

import os
import random
from datetime import datetime, timedelta
from typing import Dict

from faker import Faker

# Use simplified database layer
from db_simple import SimpleDatabase

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

def generate_model_results(is_spam) -> Dict:
    """Generate fake model prediction results"""
    # Models are generally correct but not perfect
    base_confidence = random.uniform(0.7, 0.95) if is_spam else random.uniform(0.65, 0.9)

    # Add some noise to individual models
    nb_confidence = max(0.5, min(0.99, base_confidence + random.uniform(-0.1, 0.1)))
    lr_confidence = max(0.5, min(0.99, base_confidence + random.uniform(-0.1, 0.1)))
    svc_confidence = max(0.5, min(0.99, base_confidence + random.uniform(-0.1, 0.1)))

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

def seed_database(num_records: int = 500, quiet: bool = False):
    """Seed database using simplified storage."""
    import logging
    logger = logging.getLogger(__name__)
    db = SimpleDatabase()

    if not quiet:
        print(f"🌱 Seeding database with {num_records} fake predictions...")
    else:
        logger.info(f"Seeding database with {num_records} fake predictions...")

    spam_count = 0
    ham_count = 0
    for i in range(num_records):
        is_spam = random.random() < 0.4
        if is_spam:
            msg_template = random.choice(SPAM_TEMPLATES)
            message = msg_template.format(
                amount=random.randint(100, 10000),
                amount2=random.randint(1000, 50000),
                url=f"http://{fake.domain_name()}/{fake.uri_path()}",
                phone=fake.phone_number(),
                email=fake.email(),
                product=random.choice(["IPHONE", "LAPTOP", "CAR", "VACATION", "GIFT CARD"]),
                hours=random.randint(1, 48),
                percent=random.choice([50, 60, 70, 80, 90]),
                count=random.randint(5, 99)
            )
            spam_count += 1
        else:
            msg_template = random.choice(NOT_SPAM_TEMPLATES)
            message = msg_template.format(
                time=fake.time(pattern="%I:%M %p"),
                name=fake.first_name(),
                order_id=fake.random_int(min=10000, max=99999),
                date=fake.date(pattern="%B %d")
            )
            ham_count += 1

        model_results = generate_model_results(is_spam)
        ensemble = model_results['ensemble']
        processed = message.lower()
        cluster_id = random.randint(0, 4) if is_spam else None

        # Persist single record
        db.save_prediction(
            message=message,
            prediction=ensemble['prediction'],
            confidence=ensemble['confidence'],
            is_spam=ensemble['is_spam'],
            processed_message=processed,
            model_results=model_results,
            cluster_id=cluster_id,
            timestamp=(datetime.now() - timedelta(days=random.randint(0, 30), hours=random.randint(0, 23), minutes=random.randint(0, 59))).isoformat()
        )

        if (i + 1) % 100 == 0:
            prog = f"Generated {i+1}/{num_records} predictions..."
            if not quiet:
                print("  " + prog)
            else:
                logger.info(prog)

    if not quiet:
        print("\n✅ Seeding complete")
        print("📊 Distribution:")
        print(f"   Total     : {num_records}")
        print(f"   Spam      : {spam_count} ({spam_count/num_records*100:.1f}%)")
        print(f"   Not Spam  : {ham_count} ({ham_count/num_records*100:.1f}%)")
    else:
        logger.info(f"Seeded {num_records} predictions (Spam={spam_count}, Ham={ham_count})")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Seed spam detection database")
    parser.add_argument("--count", type=int, default=500, help="Number of records to create")
    args = parser.parse_args()
    seed_database(num_records=args.count)

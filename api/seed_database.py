#!/usr/bin/env python3
"""
Seed the database with fake prediction data for testing.
Usage: python seed_database.py --count 500
"""

import random
from datetime import datetime, timedelta
from importlib import import_module
from typing import Dict, Tuple

try:
    Faker = getattr(import_module("faker"), "Faker")
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "The 'faker' package is required for seeding the database. Install it with `pip install faker`."
    ) from exc

from sql_db import SimpleDatabase

fake = Faker()

SPAM_SCENARIOS = [
    {
        "cluster_id": 0,
        "weight": 3,
        "templates": [
            "🎉 Congratulations! You've won ${amount}! Claim your reward at {url} before {deadline}.",
            "You're the lucky winner of our {product} giveaway. Confirm delivery details here: {url}.",
            "It's your day! Redeem ${amount2} in gift vouchers now ➜ {url}",
        ],
    },
    {
        "cluster_id": 1,
        "weight": 3,
        "templates": [
            "Security alert from {company}: login attempt detected in {city}. Review immediately: {url}.",
            "URGENT ⚠️ We placed a hold on your {company} account. Verify within {hours} hours: {url}",
            "[Action Required] Password reset requested from {city}. If this wasn't you confirm here: {url}.",
        ],
    },
    {
        "cluster_id": 2,
        "weight": 3,
        "templates": [
            "Bonjour, votre colis {tracking} est bloqué. Réglez {currency_symbol}{amount} de frais ici: {url}.",
            "🚚 Delivery notice: Package {tracking} requires address confirmation. Update now: {url}.",
            "Hola, necesitamos confirmar los impuestos de envío ({currency_symbol}{amount}). Paga seguro en {url}.",
        ],
    },
    {
        "cluster_id": 3,
        "weight": 2,
        "templates": [
            "Crypto insider alert: transfer {crypto_amount} {crypto} to {wallet} and double it overnight!",
            "Last call for our mining pool. Secure {crypto_amount} {crypto} allocation now ➜ {url}",
            "NFT whitelist unlocked. Send {crypto_amount} {crypto} to reserve your spot. Address: {wallet}",
        ],
    },
    {
        "cluster_id": 4,
        "weight": 2,
        "templates": [
            "IRS notice: you are due a refund of {currency_symbol}{amount}. Submit details here: {url}.",
            "Invoice {invoice} is overdue. Settle immediately via secure portal ➜ {url}",
            "Bank settlement pending. Reply to {email} with your account number for the {currency_symbol}{amount} transfer.",
        ],
    },
    {
        "cluster_id": 5,
        "weight": 1,
        "templates": [
            "Hello dear, it's {contact} from {country}. I can send you ${amount} if you help with the transfer via {url}.",
            "My heart belongs to you. Let's start our life together—please cover my flight ({currency_symbol}{amount}) at {url} ❤️",
        ],
    },
    {
        "cluster_id": 6,
        "weight": 1,
        "templates": [
            "Help {charity} raise urgent relief after the {disaster}. Donate securely: {url}",
            "We are {charity}. Families hit by {disaster} need {currency_symbol}{amount}. Give hope today ➜ {url}",
        ],
    },
    {
        "cluster_id": 7,
        "weight": 2,
        "templates": [
            "Congrats! {company} shortlisted you for {position}. Secure onboarding fee ({currency_symbol}{amount}) now: {url}",
            "Remote role paying {salary}. Submit pre-employment screening at {url} within {hours} hours.",
        ],
    },
    {
        "cluster_id": 0,
        "weight": 2,
        "templates": [
            "VIP {loyalty_program} reward worth {currency_symbol}{amount} expires {deadline}. Redeem now at {url}.",
            "{festival} giveaway! Verify your {product} win (ticket {ticket}) within {hours} hours ➜ {url}",
            "Claim your annual {loyalty_program} bonus of {currency_symbol}{amount2}. Verification code inside: {url}",
        ],
    },
    {
        "cluster_id": 1,
        "weight": 2,
        "templates": [
            "{company} flash sale: {percent}% off {product}. Activate code {voucher} before {deadline}: {url}",
            "Your {service} subscription auto-renews tonight. Manage billing ({currency_symbol}{amount}) at {url}",
            "Exclusive {industry} webinar access pending payment {currency_symbol}{amount2}. Complete registration: {url}",
        ],
    },
    {
        "cluster_id": 2,
        "weight": 2,
        "templates": [
            "{operator} notice: package {tracking} held at customs. Clear fee {currency_symbol}{amount} via {url}",
            "SMS ALERT: {service} account locked after login from {city}. Reactivate instantly: {url}",
            "Mensaje urgente: su servicio {service} se suspenderá. Use código {ticket} en {url} antes de {deadline}.",
        ],
    },
    {
        "cluster_id": 3,
        "weight": 1,
        "templates": [
            "Security scan detected {malware} on your {device}. Clean it with certified tool: {url}",
            "Critical browser patch for {platform} available. Download to avoid data loss: {url}",
            "Your cloud drive file {doc} is public. Disable the leak immediately: {url}",
        ],
    },
    {
        "cluster_id": 4,
        "weight": 2,
        "templates": [
            "Banco {bank}: bloqueamos un retiro de {currency_symbol}{amount}. Confirma identidad: {url}",
            "Investment desk {company}: wire confirmation ready for {currency_symbol}{amount2}. Upload bank slip: {url}",
            "Tax office {region}: cierre pendiente requiere pago {currency_symbol}{amount}. Liquídalo hoy: {url}",
        ],
    },
    {
        "cluster_id": 5,
        "weight": 1,
        "templates": [
            "Influencer {contact} ofrece promoción de {service}. Asegura el cupo con depósito de {currency_symbol}{amount}: {url}",
            "Limited coaching seats left. Reserve with onboarding fee {currency_symbol}{amount} ➜ {url}",
        ],
    },
    {
        "cluster_id": 6,
        "weight": 1,
        "templates": [
            "You remain subscribed to {subscription}. Confirm billing {currency_symbol}{amount} or cancel at {url}",
            "Download the marketing pack '{doc}' from {company}. Access expires {deadline}: {url}",
        ],
    },
    {
        "cluster_id": 7,
        "weight": 2,
        "templates": [
            "Domain {domain} expires tonight. Renew for {currency_symbol}{amount} at {url}",
            "SSL certificate for {domain} suspended. Restore within {hours} hours: {url}",
            "Web hosting alert: traffic spike detected across {region}. Upgrade plan at {url}",
        ],
    },
]

NOT_SPAM_SCENARIOS = [
    {
        "templates": [
            "Team,\n\nHere are the updates from today's {project} sync:\n- {task1}\n- {task2}\n\nNext standup: {date} at {schedule}.\nThanks,\n{contact}",
            "Reminder: sprint review is on {date} at {schedule}. Slide deck: {meeting_link}\nAgenda:\n• {agenda_item1}\n• {agenda_item2}",
        ],
    },
    {
        "templates": [
            "Hi {name}, your order #{order_id} is on the way! Track it here: {url}. Estimated arrival {delivery_window}.",
            "Bonjour {name}, votre colis {tracking} sera livré le {date}. Merci de rester disponible.",
        ],
    },
    {
        "templates": [
            "Family dinner at {restaurant} this Saturday around {schedule}. Let me know if {plus_one} is coming too!",
            "Grandma's birthday is {date}! We're meeting in {city} at {schedule}. Bring {item} please.",
        ],
    },
    {
        "templates": [
            "Olá {name}, segue o relatório atualizado ({doc}). Qualquer dúvida me chama. Abraços!",
            "¡Hola {name}! Adjunto el acta del proyecto {project}. Nos vemos a las {schedule} en {meeting_link}.",
        ],
    },
    {
        "templates": [
            "Great work on {project}! Highlights:\n• {highlight1}\n• {highlight2}\nLet's discuss next steps on {date}.",
            "Attaching the financial summary ({doc}). Key callouts:\n- {task1}\n- {task2}\nCheers,\n{contact}",
        ],
    },
    {
        "templates": [
            "Hi {name}, flight {tracking} departs from {airport} at {schedule}. Boarding starts 45 mins prior.",
            "Event reminder: {event} at {location} on {date}. RSVP: {url}",
        ],
    },
]


def _common_placeholders() -> Dict[str, str]:
    now = datetime.now()
    return {
        "amount": str(random.randint(200, 20000)),
        "amount2": str(random.randint(5000, 150000)),
        "url": f"https://{fake.domain_name()}/{fake.uri_path()}",
        "phone": fake.phone_number(),
        "email": fake.email(),
        "product": random.choice(["iPhone 15", "PlayStation 5", "MacBook Pro", "Tesla Test Ride", "Luxury Cruise"]),
        "hours": str(random.randint(1, 48)),
        "percent": str(random.choice([35, 40, 50, 65, 80, 90])),
        "count": str(random.randint(5, 120)),
        "company": fake.company(),
        "city": fake.city(),
        "country": fake.country(),
        "currency_symbol": random.choice(["$", "€", "£", "₦", "₹"]),
        "tracking": fake.bothify(text="PKG-####-??"),
        "crypto": random.choice(["BTC", "ETH", "SOL", "USDT", "ADA"]),
        "crypto_amount": f"{random.uniform(0.05, 2.5):.2f}",
        "wallet": fake.sha256()[:18],
        "invoice": f"INV-{random.randint(10000, 99999)}",
        "deadline": (now + timedelta(hours=random.randint(4, 36))).strftime("%b %d %I:%M %p"),
        "contact": fake.name(),
        "charity": f"{fake.last_name()} Relief Fund",
        "disaster": random.choice(["earthquake", "flood", "wildfire", "cyclone", "landslide"]),
        "position": fake.job(),
        "salary": f"{random.randint(65, 190)}k",
        "project": fake.catch_phrase(),
        "task1": fake.sentence(nb_words=8),
        "task2": fake.sentence(nb_words=9),
        "agenda_item1": fake.bs().capitalize(),
        "agenda_item2": fake.bs().capitalize(),
        "meeting_link": f"https://meet.{fake.domain_name()}/{fake.lexify(text='????')}",
        "schedule": fake.time(pattern="%I:%M %p"),
        "date": fake.date(pattern="%B %d"),
        "order_id": str(fake.random_int(min=10000, max=99999)),
        "delivery_window": fake.day_of_week() + " afternoon",
        "restaurant": fake.company() + " Bistro",
        "plus_one": fake.first_name(),
        "item": fake.word(),
        "doc": f"{fake.word().title()}_{random.randint(1, 9)}.pdf",
        "highlight1": fake.sentence(nb_words=7),
        "highlight2": fake.sentence(nb_words=7),
        "event": fake.bs().title(),
        "location": fake.city(),
        "airport": fake.city() + " International",
        "name": fake.first_name(),
        "ticket": str(random.randint(100000, 999999)),
        "loyalty_program": random.choice([
            "SkyMiles",
            "PrimeRewards",
            "UltraPoints",
            "EliteClub",
            "GalaxyPerks",
        ]),
        "festival": random.choice([
            "Summer Fest",
            "Holiday Gala",
            "Spring Carnival",
            "New Year Bash",
            "Harvest Fair",
        ]),
        "voucher": fake.bothify(text="SAVE####"),
        "operator": random.choice([
            "TeleLink",
            "GlobalTel",
            "QuickSMS",
            "MetroMobile",
            "SkyWave",
        ]),
        "service": random.choice([
            "Premium Music",
            "UltraTV",
            "Cloud Vault",
            "Prime Courier",
            "ProVPN",
        ]),
        "platform": random.choice([
            "Windows",
            "macOS",
            "Android",
            "iOS",
            "Linux",
        ]),
        "device": random.choice([
            "laptop",
            "tablet",
            "phone",
            "desktop",
            "workstation",
        ]),
        "malware": random.choice([
            "Trojan.Dropper",
            "Worm.X99",
            "Spyware.Radar",
            "RansomLock",
            "KeyLogger.Pro",
        ]),
        "bank": random.choice([
            "Banco Real",
            "Metro Bank",
            "First Capital",
            "Global Credit",
            "Union Savings",
        ]),
        "region": random.choice([
            "EMEA",
            "APAC",
            "LATAM",
            "North America",
            "Scandinavia",
        ]),
        "subscription": random.choice([
            "Growth Digest",
            "Sales Pulse",
            "DevOps Weekly",
            "UI Trends",
            "Founder Notes",
        ]),
        "industry": random.choice([
            "fintech",
            "healthtech",
            "edtech",
            "ecommerce",
            "cybersecurity",
        ]),
        "domain": fake.domain_name(),
    }


def _render_template(template: str, extra: Dict[str, str] | None = None) -> str:
    placeholders = _common_placeholders()
    if extra:
        placeholders.update(extra)
    return template.format(**placeholders)


def generate_spam_message() -> Tuple[str, int]:
    """Generate a fake spam message with cluster association."""
    scenario = random.choices(SPAM_SCENARIOS, weights=[s["weight"] for s in SPAM_SCENARIOS], k=1)[0]
    template = random.choice(scenario["templates"])
    extra = scenario.get("extra")
    if callable(extra):
        extra = extra()
    message = _render_template(template, extra)
    return message, scenario.get("cluster_id", random.randint(0, 7))


def generate_not_spam_message() -> str:
    """Generate a fake legitimate message."""
    scenario = random.choice(NOT_SPAM_SCENARIOS)
    template = random.choice(scenario["templates"])
    extra = scenario.get("extra")
    if callable(extra):
        extra = extra()
    return _render_template(template, extra)

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
def generate_timestamp() -> str:
    """Create a timestamp distributed across the year with realistic patterns."""
    now = datetime.now()
    start_of_year = datetime(now.year, 1, 1)
    
    # Calculate days from start of year to now
    days_elapsed = (now - start_of_year).days
    
    # Weighted distribution to create interesting patterns
    # 20% recent (last 7 days), 30% last month, 50% rest of year
    rand = random.random()
    if rand < 0.20:
        # Recent data - last 7 days
        day_offset = random.uniform(0, 7)
    elif rand < 0.50:
        # Last month
        day_offset = random.uniform(7, 30)
    else:
        # Distributed across the rest of the year
        day_offset = random.uniform(30, days_elapsed)
    
    base = now - timedelta(days=day_offset)
    
    # Vary time of day - business hours more likely
    if random.random() < 0.65:
        hour = random.randint(8, 20)
    else:
        hour = random.randint(0, 23)
    
    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    
    base = base.replace(hour=hour, minute=minute, second=second, microsecond=0)
    
    # Create some clustering on certain dates (simulating spam campaigns)
    # 15% chance to cluster around a "campaign date"
    if random.random() < 0.15:
        campaign_offset = random.randint(0, days_elapsed)
        campaign_date = start_of_year + timedelta(days=campaign_offset)
        # Cluster within +/- 2 days of campaign
        cluster_offset = random.uniform(-2, 2)
        base = campaign_date + timedelta(days=cluster_offset, hours=hour, minutes=minute, seconds=second)
    
    # Ensure we don't create future timestamps
    if base > now:
        base = now - timedelta(minutes=random.randint(1, 60))
    
    # Ensure we stay within current year
    if base < start_of_year:
        base = start_of_year + timedelta(hours=random.randint(0, 23), minutes=random.randint(0, 59))
    
    return base.isoformat()


def seed_database(num_records: int = 500, quiet: bool = False):
    """Seed database using simplified storage."""
    import logging
    logger = logging.getLogger(__name__)
    db = SimpleDatabase()

    if not quiet:
        print(f"Seeding database with {num_records} fake predictions...")
    else:
        logger.info(f"Seeding database with {num_records} fake predictions...")

    spam_count = 0
    ham_count = 0
    for i in range(num_records):
        is_spam = random.random() < 0.45
        if is_spam:
            message, cluster_id = generate_spam_message()
            spam_count += 1
        else:
            message = generate_not_spam_message()
            ham_count += 1
            cluster_id = None

        model_results = generate_model_results(is_spam)
        ensemble = model_results['ensemble']
        processed = message.lower()

        # Persist single record
        db.save_prediction(
            message=message,
            prediction=ensemble['prediction'],
            confidence=ensemble['confidence'],
            is_spam=ensemble['is_spam'],
            processed_message=processed,
            model_results=model_results,
            cluster_id=cluster_id,
            timestamp=generate_timestamp()
        )

        if (i + 1) % 100 == 0:
            prog = f"Generated {i+1}/{num_records} predictions..."
            if not quiet:
                print("  " + prog)
            else:
                logger.info(prog)

    if not quiet:
        print("\nSeeding complete")
        print("Distribution:")
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

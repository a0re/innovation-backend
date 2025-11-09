"""
Cluster naming and categorization for spam detection.
Maps cluster IDs to human-readable names and descriptions.
"""

CLUSTER_NAMES = {
    0: {
        "name": "Prize & Sweepstakes",
        "short_name": "Prize Scams",
        "description": "Messages about winning prizes, cash rewards, or urgent claims",
        "icon": "🎁",
        "color": "#f59e0b"  # amber
    },
    1: {
        "name": "General Commercial",
        "short_name": "Commercial",
        "description": "General marketing and promotional messages",
        "icon": "🏪",
        "color": "#3b82f6"  # blue
    },
    2: {
        "name": "SMS Spam",
        "short_name": "SMS",
        "description": "Mobile text message spam and ringtones",
        "icon": "📱",
        "color": "#8b5cf6"  # purple
    },
    3: {
        "name": "URL & Links",
        "short_name": "URLs",
        "description": "Messages with suspicious links and website URLs",
        "icon": "🔗",
        "color": "#06b6d4"  # cyan
    },
    4: {
        "name": "Financial Fraud",
        "short_name": "Financial",
        "description": "Financial scams, business opportunities, and money transfers",
        "icon": "💰",
        "color": "#10b981"  # green
    },
    5: {
        "name": "Aggressive Marketing",
        "short_name": "Marketing",
        "description": "Pushy sales tactics and free offers",
        "icon": "📢",
        "color": "#ec4899"  # pink
    },
    6: {
        "name": "Email Lists",
        "short_name": "Mailing Lists",
        "description": "Unsolicited mailing list and subscription spam",
        "icon": "📧",
        "color": "#6366f1"  # indigo
    },
    7: {
        "name": "Website Spam",
        "short_name": "Websites",
        "description": "Web hosting, domain, and website-related spam",
        "icon": "🌐",
        "color": "#14b8a6"  # teal
    }
}


def get_cluster_name(cluster_id: int) -> str:
    """Get the full name for a cluster ID."""
    cluster = CLUSTER_NAMES.get(cluster_id)
    return cluster["name"] if cluster else f"Cluster {cluster_id}"


def get_cluster_short_name(cluster_id: int) -> str:
    """Get the short name for a cluster ID."""
    cluster = CLUSTER_NAMES.get(cluster_id)
    return cluster["short_name"] if cluster else f"Cluster {cluster_id}"


def get_cluster_description(cluster_id: int) -> str:
    """Get the description for a cluster ID."""
    cluster = CLUSTER_NAMES.get(cluster_id)
    return cluster["description"] if cluster else "Unknown cluster type"


def get_cluster_info(cluster_id: int) -> dict:
    """Get all information for a cluster ID."""
    cluster = CLUSTER_NAMES.get(cluster_id)
    if cluster:
        return {
            "cluster_id": cluster_id,
            "name": cluster["name"],
            "short_name": cluster["short_name"],
            "description": cluster["description"],
            "icon": cluster["icon"],
            "color": cluster["color"]
        }
    return {
        "cluster_id": cluster_id,
        "name": f"Cluster {cluster_id}",
        "short_name": f"Cluster {cluster_id}",
        "description": "Unknown cluster type",
        "icon": "❓",
        "color": "#6b7280"  # gray
    }


def get_all_cluster_names() -> dict:
    """Get all cluster names as a dictionary mapping cluster_id to name."""
    return {cluster_id: info["name"] for cluster_id, info in CLUSTER_NAMES.items()}

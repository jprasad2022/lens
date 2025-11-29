"""
Calculate REAL metrics from actual MovieLens data
No fake numbers - only actual calculations
"""

import os
import sys
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import json

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent / 'backend'))


def calculate_gini_coefficient(values: List[float]) -> float:
    """Calculate real Gini coefficient"""
    sorted_values = sorted(values)
    n = len(values)
    if n == 0 or sum(values) == 0:
        return 0
    
    cumsum = 0
    for i, value in enumerate(sorted_values):
        cumsum += (n - i) * value
    
    return (n + 1 - 2 * cumsum / sum(sorted_values)) / n


def analyze_ratings_file(filepath: str) -> Dict:
    """Analyze the actual ratings.dat file"""
    movie_ratings = defaultdict(list)
    user_ratings = defaultdict(list)
    total_ratings = 0
    
    print(f"Reading ratings from: {filepath}")
    
    try:
        with open(filepath, 'r', encoding='latin-1') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Parse format: UserID::MovieID::Rating::Timestamp
                parts = line.split('::')
                if len(parts) == 4:
                    user_id = int(parts[0])
                    movie_id = int(parts[1])
                    rating = float(parts[2])
                    
                    movie_ratings[movie_id].append(rating)
                    user_ratings[user_id].append(rating)
                    total_ratings += 1
    
    except FileNotFoundError:
        print(f"ERROR: Could not find ratings file at {filepath}")
        return None
    
    # Calculate movie statistics
    movie_counts = {mid: len(ratings) for mid, ratings in movie_ratings.items()}
    rating_counts = list(movie_counts.values())
    
    # Calculate REAL Gini coefficient
    gini = calculate_gini_coefficient(rating_counts)
    
    # Calculate REAL percentiles
    sorted_counts = sorted(rating_counts, reverse=True)
    total_movies = len(sorted_counts)
    
    # Movies with <= 100 ratings
    movies_100_or_less = sum(1 for count in rating_counts if count <= 100)
    percent_100_or_less = (movies_100_or_less / total_movies) * 100
    
    # Top 10% coverage
    top_10_percent_count = int(total_movies * 0.1)
    top_10_percent_ratings = sum(sorted_counts[:top_10_percent_count])
    top_10_percent_coverage = (top_10_percent_ratings / total_ratings) * 100
    
    # Bottom 50% coverage
    bottom_50_percent_count = int(total_movies * 0.5)
    bottom_50_percent_ratings = sum(sorted_counts[-bottom_50_percent_count:])
    bottom_50_percent_coverage = (bottom_50_percent_ratings / total_ratings) * 100
    
    return {
        "total_movies": total_movies,
        "total_ratings": total_ratings,
        "total_users": len(user_ratings),
        "gini_coefficient": round(gini, 3),
        "movies_with_100_or_less_ratings": movies_100_or_less,
        "percent_movies_with_100_or_less": round(percent_100_or_less, 1),
        "top_10_percent_coverage": round(top_10_percent_coverage, 1),
        "bottom_50_percent_coverage": round(bottom_50_percent_coverage, 1),
        "avg_ratings_per_movie": round(total_ratings / total_movies, 1),
        "avg_ratings_per_user": round(total_ratings / len(user_ratings), 1)
    }


def analyze_user_demographics(filepath: str) -> Dict:
    """Analyze the actual users.dat file"""
    gender_count = {'M': 0, 'F': 0}
    age_groups = defaultdict(int)
    occupations = defaultdict(int)
    total_users = 0
    
    print(f"Reading user demographics from: {filepath}")
    
    try:
        with open(filepath, 'r', encoding='latin-1') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Parse format: UserID::Gender::Age::Occupation::ZipCode
                parts = line.split('::')
                if len(parts) == 5:
                    gender = parts[1]
                    age = int(parts[2])
                    occupation = int(parts[3])
                    
                    gender_count[gender] = gender_count.get(gender, 0) + 1
                    
                    # Age grouping
                    if age < 18:
                        age_groups['<18'] += 1
                    elif age < 25:
                        age_groups['18-24'] += 1
                    elif age < 35:
                        age_groups['25-34'] += 1
                    elif age < 45:
                        age_groups['35-44'] += 1
                    elif age < 55:
                        age_groups['45-54'] += 1
                    else:
                        age_groups['55+'] += 1
                    
                    occupations[occupation] += 1
                    total_users += 1
    
    except FileNotFoundError:
        print(f"ERROR: Could not find users file at {filepath}")
        return None
    
    return {
        "total_users": total_users,
        "gender_distribution": {
            "male": gender_count.get('M', 0),
            "female": gender_count.get('F', 0),
            "male_percentage": round(gender_count.get('M', 0) / total_users * 100, 1),
            "female_percentage": round(gender_count.get('F', 0) / total_users * 100, 1)
        },
        "age_distribution": dict(age_groups),
        "age_percentages": {
            group: round(count / total_users * 100, 1) 
            for group, count in age_groups.items()
        }
    }


def analyze_kafka_logs() -> Dict:
    """Analyze actual Kafka logs if available"""
    # Check if there are any actual log files
    log_dir = Path("logs")
    kafka_logs = []
    
    if log_dir.exists():
        kafka_logs = list(log_dir.glob("*kafka*.log")) + list(log_dir.glob("*event*.log"))
    
    if kafka_logs:
        print(f"Found {len(kafka_logs)} Kafka log files")
        # Would parse actual logs here
        return {"status": "logs_found", "count": len(kafka_logs)}
    else:
        print("No Kafka logs found - system may not have been run yet")
        return {"status": "no_logs", "count": 0}


def main():
    """Calculate all real metrics"""
    print("=" * 60)
    print("CALCULATING REAL METRICS FROM ACTUAL DATA")
    print("=" * 60)
    
    # Paths to actual data files
    data_dir = Path("data")
    ratings_file = data_dir / "ratings.dat"
    users_file = data_dir / "users.dat"
    
    results = {}
    
    # 1. Analyze ratings for popularity bias
    print("\n1. ANALYZING RATINGS DATA...")
    if ratings_file.exists():
        ratings_analysis = analyze_ratings_file(str(ratings_file))
        if ratings_analysis:
            results['ratings_analysis'] = ratings_analysis
            print(f"\nREAL Popularity Bias Metrics:")
            print(f"  - Gini Coefficient: {ratings_analysis['gini_coefficient']}")
            print(f"  - Total Movies: {ratings_analysis['total_movies']:,}")
            print(f"  - Total Ratings: {ratings_analysis['total_ratings']:,}")
            print(f"  - Movies with ≤100 ratings: {ratings_analysis['percent_movies_with_100_or_less']}%")
            print(f"  - Top 10% movies have {ratings_analysis['top_10_percent_coverage']}% of ratings")
            print(f"  - Bottom 50% movies have {ratings_analysis['bottom_50_percent_coverage']}% of ratings")
    else:
        print(f"ERROR: Ratings file not found at {ratings_file}")
    
    # 2. Analyze user demographics
    print("\n2. ANALYZING USER DEMOGRAPHICS...")
    if users_file.exists():
        user_analysis = analyze_user_demographics(str(users_file))
        if user_analysis:
            results['user_demographics'] = user_analysis
            print(f"\nREAL User Demographics:")
            print(f"  - Total Users: {user_analysis['total_users']:,}")
            print(f"  - Gender Distribution:")
            print(f"    - Male: {user_analysis['gender_distribution']['male_percentage']}%")
            print(f"    - Female: {user_analysis['gender_distribution']['female_percentage']}%")
            print(f"  - Age Distribution:")
            for group, percent in user_analysis['age_percentages'].items():
                print(f"    - {group}: {percent}%")
    else:
        print(f"ERROR: Users file not found at {users_file}")
    
    # 3. Check for actual Kafka logs
    print("\n3. CHECKING FOR KAFKA LOGS...")
    kafka_analysis = analyze_kafka_logs()
    results['kafka_logs'] = kafka_analysis
    
    # 4. Note what we CANNOT calculate without running the system
    print("\n4. METRICS REQUIRING LIVE SYSTEM:")
    print("  - Per-demographic RMSE (needs model predictions)")
    print("  - Actual recommendation distributions (needs recommendation logs)")
    print("  - Feedback loop trends (needs historical data)")
    print("  - Security violations (needs runtime logs)")
    print("  - A/B test results (needs experiment data)")
    
    # Save results
    output_file = "real_metrics_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Real metrics saved to {output_file}")
    
    print("\n" + "=" * 60)
    print("USE THESE REAL NUMBERS IN YOUR REPORT!")
    print("=" * 60)


if __name__ == "__main__":
    main()
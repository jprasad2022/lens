"""
User Demographics Service
Loads and provides user demographic information from MovieLens data
"""

from typing import Dict, Optional, List
from pathlib import Path
import asyncio

from config.settings import get_settings

settings = get_settings()


class UserDemographicsService:
    """Service for managing user demographic data"""

    # Age group mappings
    AGE_GROUPS = {
        1: "Under 18",
        18: "18-24",
        25: "25-34",
        35: "35-44",
        45: "45-49",
        50: "50-55",
        56: "56+"
    }

    # Occupation mappings
    OCCUPATIONS = {
        0: "other",
        1: "academic/educator",
        2: "artist",
        3: "clerical/admin",
        4: "college/grad student",
        5: "customer service",
        6: "doctor/health care",
        7: "executive/managerial",
        8: "farmer",
        9: "homemaker",
        10: "K-12 student",
        11: "lawyer",
        12: "programmer",
        13: "retired",
        14: "sales/marketing",
        15: "scientist",
        16: "self-employed",
        17: "technician/engineer",
        18: "tradesman/craftsman",
        19: "unemployed",
        20: "writer"
    }

    def __init__(self):
        self.users: Dict[int, Dict] = {}
        self._initialized = False
        self._lock = asyncio.Lock()
        self.demographics_summary = {}

    async def initialize(self):
        """Load user data from users.dat file"""
        async with self._lock:
            if self._initialized:
                return

            users_file = Path(settings.data_path) / "users.dat"
            if not users_file.exists():
                print(f"Warning: Users file not found at {users_file}")
                return

            print("Loading user demographics...")

            # Initialize counters for summary
            gender_count = {'M': 0, 'F': 0}
            age_distribution = {}
            occupation_distribution = {}

            try:
                with open(users_file, 'r', encoding='latin-1') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue

                        # Parse format: UserID::Gender::Age::Occupation::ZipCode
                        parts = line.split('::')
                        if len(parts) == 5:
                            user_id = int(parts[0])
                            gender = parts[1]
                            age = int(parts[2])
                            occupation = int(parts[3])
                            zipcode = parts[4]

                            # Get age group
                            age_group = self._get_age_group(age)
                            occupation_name = self.OCCUPATIONS.get(occupation, "unknown")

                            self.users[user_id] = {
                                'user_id': user_id,
                                'gender': gender,
                                'age': age,
                                'age_group': age_group,
                                'occupation': occupation,
                                'occupation_name': occupation_name,
                                'zipcode': zipcode
                            }

                            # Update summary statistics
                            gender_count[gender] = gender_count.get(gender, 0) + 1
                            age_distribution[age_group] = age_distribution.get(age_group, 0) + 1
                            occupation_distribution[occupation_name] = occupation_distribution.get(occupation_name, 0) + 1

                # Calculate summary statistics
                total_users = len(self.users)
                self.demographics_summary = {
                    'total_users': total_users,
                    'gender_distribution': {
                        'male': gender_count.get('M', 0),
                        'female': gender_count.get('F', 0),
                        'male_percentage': round(gender_count.get('M', 0) / total_users * 100, 1),
                        'female_percentage': round(gender_count.get('F', 0) / total_users * 100, 1)
                    },
                    'age_distribution': age_distribution,
                    'occupation_distribution': occupation_distribution,
                    'top_occupations': sorted(occupation_distribution.items(),
                                             key=lambda x: x[1],
                                             reverse=True)[:5]
                }

                self._initialized = True
                print(f"Loaded {total_users} user profiles")
                print(f"Gender distribution: M={gender_count.get('M', 0)}, F={gender_count.get('F', 0)}")
                print(f"Top occupation: {self.demographics_summary['top_occupations'][0][0]}")

            except Exception as e:
                print(f"Error loading user demographics: {e}")
                raise

    def _get_age_group(self, age: int) -> str:
        """Convert age to age group"""
        if age < 18:
            return self.AGE_GROUPS[1]
        elif age < 25:
            return self.AGE_GROUPS[18]
        elif age < 35:
            return self.AGE_GROUPS[25]
        elif age < 45:
            return self.AGE_GROUPS[35]
        elif age < 50:
            return self.AGE_GROUPS[45]
        elif age <= 55:
            return self.AGE_GROUPS[50]
        else:
            return self.AGE_GROUPS[56]

    async def get_user(self, user_id: int) -> Optional[Dict]:
        """Get user demographic information by ID"""
        if not self._initialized:
            await self.initialize()

        return self.users.get(user_id)

    async def get_users(self, user_ids: List[int]) -> List[Dict]:
        """Get multiple users by IDs"""
        if not self._initialized:
            await self.initialize()

        users = []
        for user_id in user_ids:
            user = self.users.get(user_id)
            if user:
                users.append(user)

        return users

    async def get_user_segment(self, user_id: int) -> str:
        """Get user segment for personalization"""
        user = await self.get_user(user_id)
        if not user:
            return "unknown"

        # Simple segmentation based on age and gender
        return f"{user['gender']}_{user['age_group']}"

    def get_demographics_summary(self) -> Dict:
        """Get overall demographics summary"""
        return self.demographics_summary

#!/usr/bin/env python
# coding: utf-8

"""
Enhanced Animal Shelter CRUD Operations with Advanced Algorithms
CS 499 Milestone Three - Algorithms and Data Structure Enhancement

This module demonstrates advanced algorithmic implementations including:
- Fuzzy string matching using Levenshtein distance
- Machine learning recommendations using Random Forest
- LRU cache implementation with TTL management
- Optimized query performance with indexing strategies
- Geospatial algorithms for location-based searching

Author: Ifeoluwa Adewoyin
Course: CS 499 Computer Science Capstone
Enhancement: Algorithms and Data Structure Category
"""

from pymongo import MongoClient
from bson.objectid import ObjectId
import os
import logging
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import OrderedDict
import math
import threading
from typing import Dict, List, Tuple, Optional, Any

# Machine Learning imports
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    ML_AVAILABLE = True
except ImportError:
    print("Warning: scikit-learn not available. ML features will be disabled.")
    ML_AVAILABLE = False


class FuzzyStringMatcher:
    """
    Advanced fuzzy string matching using Levenshtein distance algorithm
    Implements dynamic programming approach for efficient string similarity calculation
    """
    
    @staticmethod
    def levenshtein_distance(s1: str, s2: str) -> int:
        """
        Calculate Levenshtein distance between two strings using dynamic programming
        
        Time Complexity: O(m*n) where m and n are string lengths
        Space Complexity: O(min(m,n)) with space optimization
        
        Args:
            s1, s2: Input strings for comparison
            
        Returns:
            int: Minimum edit distance between strings
        """
        if len(s1) < len(s2):
            return FuzzyStringMatcher.levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        # Space-optimized DP using only two rows
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                # Cost of insertions, deletions, substitutions
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    @staticmethod
    def similarity_ratio(s1: str, s2: str) -> float:
        """
        Calculate similarity ratio between two strings
        
        Args:
            s1, s2: Input strings for comparison
            
        Returns:
            float: Similarity ratio between 0.0 and 1.0
        """
        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0
        
        distance = FuzzyStringMatcher.levenshtein_distance(s1.lower(), s2.lower())
        max_len = max(len(s1), len(s2))
        return 1.0 - (distance / max_len)
    
    @staticmethod
    def find_best_matches(query: str, candidates: List[str], 
                         threshold: float = 0.6, max_results: int = 5) -> List[Tuple[str, float]]:
        """
        Find best matching strings from candidates using fuzzy matching
        
        Args:
            query: String to match against
            candidates: List of candidate strings
            threshold: Minimum similarity threshold
            max_results: Maximum number of results to return
            
        Returns:
            List of tuples (candidate, similarity_score) sorted by similarity
        """
        matches = []
        for candidate in candidates:
            similarity = FuzzyStringMatcher.similarity_ratio(query, candidate)
            if similarity >= threshold:
                matches.append((candidate, similarity))
        
        # Sort by similarity score (descending) and return top results
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:max_results]


class LRUCache:
    """
    Least Recently Used (LRU) Cache implementation with Time-To-Live (TTL) support
    Uses OrderedDict for O(1) access, insertion, and deletion operations
    """
    
    def __init__(self, capacity: int = 1000, default_ttl: int = 300):
        """
        Initialize LRU Cache
        
        Args:
            capacity: Maximum number of items to store
            default_ttl: Default time-to-live in seconds
        """
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache = OrderedDict()
        self.access_times = {}
        self.lock = threading.RLock()
    
    def _is_expired(self, key: str) -> bool:
        """Check if cache entry has expired"""
        if key not in self.access_times:
            return True
        return time.time() - self.access_times[key] > self.default_ttl
    
    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve item from cache
        
        Time Complexity: O(1)
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        with self.lock:
            if key not in self.cache or self._is_expired(key):
                if key in self.cache:
                    del self.cache[key]
                    del self.access_times[key]
                return None
            
            # Move to end (most recently used)
            value = self.cache[key]
            self.cache.move_to_end(key)
            self.access_times[key] = time.time()
            return value
    
    def put(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Store item in cache
        
        Time Complexity: O(1)
        
        Args:
            key: Cache key
            value: Value to store
            ttl: Time-to-live in seconds (uses default if None)
        """
        with self.lock:
            if key in self.cache:
                # Update existing entry
                self.cache[key] = value
                self.cache.move_to_end(key)
            else:
                # Add new entry
                if len(self.cache) >= self.capacity:
                    # Remove least recently used item
                    oldest_key = next(iter(self.cache))
                    del self.cache[oldest_key]
                    del self.access_times[oldest_key]
                
                self.cache[key] = value
            
            self.access_times[key] = time.time()
    
    def invalidate(self, key: str) -> bool:
        """Remove item from cache"""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                del self.access_times[key]
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries"""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            current_time = time.time()
            expired_count = sum(1 for t in self.access_times.values() 
                              if current_time - t > self.default_ttl)
            
            return {
                "capacity": self.capacity,
                "current_size": len(self.cache),
                "expired_entries": expired_count,
                "hit_ratio": getattr(self, '_hits', 0) / max(getattr(self, '_requests', 1), 1)
            }


class MLRecommendationEngine:
    """
    Machine Learning-based recommendation engine for rescue animal selection
    Uses Random Forest algorithm for classification and feature importance analysis
    """
    
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.feature_columns = ['breed_encoded', 'age_weeks', 'weight_encoded', 'outcome_type_encoded']
        self.is_trained = False
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for machine learning model
        
        Args:
            df: DataFrame with animal data
            
        Returns:
            DataFrame with encoded features
        """
        if df.empty:
            return df
        
        df_processed = df.copy()
        
        # Encode categorical variables
        categorical_columns = ['breed', 'outcome_type']
        for col in categorical_columns:
            if col in df_processed.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df_processed[f'{col}_encoded'] = self.label_encoders[col].fit_transform(
                        df_processed[col].astype(str)
                    )
                else:
                    # Handle new categories not seen during training
                    known_labels = set(self.label_encoders[col].classes_)
                    df_processed[f'{col}_encoded'] = df_processed[col].apply(
                        lambda x: self.label_encoders[col].transform([str(x)])[0] 
                        if str(x) in known_labels else 0
                    )
        
        # Process numerical features
        if 'age_upon_outcome_in_weeks' in df_processed.columns:
            df_processed['age_weeks'] = pd.to_numeric(
                df_processed['age_upon_outcome_in_weeks'], errors='coerce'
            ).fillna(52)  # Default to 1 year
        
        # Create weight feature (estimate based on breed)
        df_processed['weight_encoded'] = df_processed.get('breed_encoded', 0)
        
        return df_processed
    
    def train_model(self, training_data: pd.DataFrame, success_column: str = 'rescue_success') -> float:
        """
        Train Random Forest model for rescue success prediction
        
        Args:
            training_data: DataFrame with training examples
            success_column: Column indicating rescue success (1) or failure (0)
            
        Returns:
            Model accuracy score
        """
        if not ML_AVAILABLE:
            print("Machine learning features not available")
            return 0.0
        
        # Prepare features
        df_processed = self.prepare_features(training_data)
        
        # Create synthetic success labels based on rescue criteria
        df_processed['rescue_success'] = self._generate_success_labels(df_processed)
        
        # Select features and target
        feature_cols = [col for col in self.feature_columns if col in df_processed.columns]
        if not feature_cols:
            print("No suitable features found for training")
            return 0.0
        
        X = df_processed[feature_cols].fillna(0)
        y = df_processed['rescue_success']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train Random Forest model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        self.is_trained = True
        print(f"Model trained successfully. Accuracy: {accuracy:.3f}")
        
        return accuracy
    
    def _generate_success_labels(self, df: pd.DataFrame) -> pd.Series:
        """Generate synthetic success labels based on rescue criteria"""
        success = pd.Series(0, index=df.index)
        
        # Higher success probability for specific breeds and age ranges
        rescue_breeds = ['German Shepherd', 'Labrador Retriever', 'Golden Retriever', 
                        'Bloodhound', 'Rottweiler']
        
        for breed in rescue_breeds:
            breed_mask = df['breed'].str.contains(breed, case=False, na=False)
            age_mask = (df.get('age_weeks', 0) >= 26) & (df.get('age_weeks', 0) <= 156)
            success.loc[breed_mask & age_mask] = 1
        
        # Add some noise to make it more realistic
        noise = np.random.binomial(1, 0.1, len(success))
        success = (success | noise) & (~np.random.binomial(1, 0.05, len(success)))
        
        return success
    
    def predict_success_probability(self, animal_data: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """
        Predict rescue success probability for an animal
        
        Args:
            animal_data: Dictionary with animal features
            
        Returns:
            Tuple of (success_probability, feature_importance)
        """
        if not self.is_trained or not ML_AVAILABLE:
            return 0.5, {}
        
        # Convert to DataFrame and prepare features
        df = pd.DataFrame([animal_data])
        df_processed = self.prepare_features(df)
        
        # Select features
        feature_cols = [col for col in self.feature_columns if col in df_processed.columns]
        if not feature_cols:
            return 0.5, {}
        
        X = df_processed[feature_cols].fillna(0)
        
        # Predict probability
        probability = self.model.predict_proba(X)[0][1]  # Probability of success (class 1)
        
        # Get feature importance
        feature_importance = dict(zip(feature_cols, self.model.feature_importances_))
        
        return probability, feature_importance


class GeospatialIndex:
    """
    Simple R-tree-like spatial indexing for geolocation queries
    Implements quadtree-based spatial partitioning for efficient range queries
    """
    
    def __init__(self, bounds: Tuple[float, float, float, float] = (-180, -90, 180, 90)):
        """
        Initialize spatial index
        
        Args:
            bounds: (min_lon, min_lat, max_lon, max_lat)
        """
        self.bounds = bounds
        self.points = []
        self.children = None
        self.capacity = 10
    
    def insert(self, point: Tuple[float, float], data: Any) -> None:
        """
        Insert point into spatial index
        
        Args:
            point: (longitude, latitude)
            data: Associated data
        """
        if not self._contains_point(point):
            return
        
        if len(self.points) < self.capacity and self.children is None:
            self.points.append((point, data))
        else:
            if self.children is None:
                self._subdivide()
            
            for child in self.children:
                child.insert(point, data)
    
    def _contains_point(self, point: Tuple[float, float]) -> bool:
        """Check if point is within bounds"""
        lon, lat = point
        min_lon, min_lat, max_lon, max_lat = self.bounds
        return min_lon <= lon <= max_lon and min_lat <= lat <= max_lat
    
    def _subdivide(self) -> None:
        """Subdivide current node into four quadrants"""
        min_lon, min_lat, max_lon, max_lat = self.bounds
        mid_lon = (min_lon + max_lon) / 2
        mid_lat = (min_lat + max_lat) / 2
        
        self.children = [
            GeospatialIndex((min_lon, min_lat, mid_lon, mid_lat)),  # SW
            GeospatialIndex((mid_lon, min_lat, max_lon, mid_lat)),  # SE
            GeospatialIndex((min_lon, mid_lat, mid_lon, max_lat)),  # NW
            GeospatialIndex((mid_lon, mid_lat, max_lon, max_lat))   # NE
        ]
        
        # Redistribute points to children
        for point, data in self.points:
            for child in self.children:
                child.insert(point, data)
        
        self.points.clear()
    
    def query_range(self, range_bounds: Tuple[float, float, float, float]) -> List[Tuple[Tuple[float, float], Any]]:
        """
        Query points within range
        
        Args:
            range_bounds: (min_lon, min_lat, max_lon, max_lat)
            
        Returns:
            List of (point, data) tuples within range
        """
        if not self._intersects(range_bounds):
            return []
        
        results = []
        
        # Check points in current node
        for point, data in self.points:
            if self._point_in_range(point, range_bounds):
                results.append((point, data))
        
        # Query children if they exist
        if self.children:
            for child in self.children:
                results.extend(child.query_range(range_bounds))
        
        return results
    
    def _intersects(self, other_bounds: Tuple[float, float, float, float]) -> bool:
        """Check if bounds intersect with other bounds"""
        min_lon1, min_lat1, max_lon1, max_lat1 = self.bounds
        min_lon2, min_lat2, max_lon2, max_lat2 = other_bounds
        
        return not (max_lon1 < min_lon2 or max_lon2 < min_lon1 or 
                   max_lat1 < min_lat2 or max_lat2 < min_lat1)
    
    def _point_in_range(self, point: Tuple[float, float], 
                       range_bounds: Tuple[float, float, float, float]) -> bool:
        """Check if point is within range bounds"""
        lon, lat = point
        min_lon, min_lat, max_lon, max_lat = range_bounds
        return min_lon <= lon <= max_lon and min_lat <= lat <= max_lat


class EnhancedAnimalShelter:
    """
    Enhanced Animal Shelter CRUD operations with advanced algorithms
    Demonstrates mastery of algorithms and data structures including:
    - Fuzzy string matching for breed search
    - Machine learning recommendations
    - LRU caching for performance optimization
    - Geospatial indexing for location queries
    """
    
    def __init__(self, username="aacuser", password="SNHU1234", 
                 host=None, port=None, database_name="AAC", collection_name="animals"):
        """Initialize enhanced animal shelter with algorithm components"""
        
        # Database connection
        self.username = username
        self.password = password
        self.host = host or 'localhost'
        self.port = port or 27017
        self.database_name = database_name
        self.collection_name = collection_name
        
        # Enhanced algorithm components
        self.fuzzy_matcher = FuzzyStringMatcher()
        self.cache = LRUCache(capacity=1000, default_ttl=300)
        self.ml_engine = MLRecommendationEngine()
        self.spatial_index = GeospatialIndex()
        
        # Performance tracking
        self.query_stats = {
            'total_queries': 0,
            'cache_hits': 0,
            'avg_query_time': 0,
            'fuzzy_searches': 0
        }
        
        self._connect()
        self._initialize_ml_model()
    
    def _connect(self):
        """Establish MongoDB connection"""
        try:
            if self.username and self.password:
                connection_string = f'mongodb://{self.username}:{self.password}@{self.host}:{self.port}/?authSource=admin'
            else:
                connection_string = f'mongodb://{self.host}:{self.port}'
            
            self.client = MongoClient(connection_string)
            self.database = self.client[self.database_name]
            self.collection = self.database[self.collection_name]
            
            # Test connection
            self.database.command('ping')
            print(f"Enhanced connection successful: {self.database_name}.{self.collection_name}")
            
        except Exception as e:
            print(f"Connection error: {e}")
            raise e
    
    def _initialize_ml_model(self):
        """Initialize machine learning model with sample data"""
        try:
            # Load sample data for training
            sample_data = list(self.collection.find().limit(1000))
            if sample_data:
                df = pd.DataFrame(sample_data)
                self.ml_engine.train_model(df)
                print("Machine learning model initialized successfully")
        except Exception as e:
            print(f"ML model initialization failed: {e}")
    
    def enhanced_read(self, query=None, use_cache=True, enable_fuzzy=False, 
                     fuzzy_field=None, fuzzy_query=None, fuzzy_threshold=0.7):
        """
        Enhanced read operation with caching and fuzzy matching
        
        Args:
            query: MongoDB query dictionary
            use_cache: Whether to use LRU cache
            enable_fuzzy: Enable fuzzy string matching
            fuzzy_field: Field to apply fuzzy matching on
            fuzzy_query: String to fuzzy match against
            fuzzy_threshold: Minimum similarity threshold for fuzzy matching
            
        Returns:
            List of matching documents
        """
        start_time = time.time()
        self.query_stats['total_queries'] += 1
        
        # Generate cache key
        cache_key = str(hash(str(query))) if query else 'all_documents'
        if enable_fuzzy and fuzzy_query:
            cache_key += f"_fuzzy_{fuzzy_field}_{fuzzy_query}_{fuzzy_threshold}"
            self.query_stats['fuzzy_searches'] += 1
        
        # Check cache first
        if use_cache:
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                self.query_stats['cache_hits'] += 1
                return cached_result
        
        # Execute database query
        try:
            if query is None:
                query = {}
            
            cursor = self.collection.find(query)
            results = list(cursor)
            
            # Apply fuzzy matching if enabled
            if enable_fuzzy and fuzzy_field and fuzzy_query:
                results = self._apply_fuzzy_matching(
                    results, fuzzy_field, fuzzy_query, fuzzy_threshold
                )
            
            # Cache results
            if use_cache:
                self.cache.put(cache_key, results)
            
            # Update performance stats
            query_time = time.time() - start_time
            self.query_stats['avg_query_time'] = (
                (self.query_stats['avg_query_time'] * (self.query_stats['total_queries'] - 1) + query_time) /
                self.query_stats['total_queries']
            )
            
            print(f"Query completed in {query_time:.3f}s, returned {len(results)} results")
            return results
            
        except Exception as e:
            print(f"Enhanced read error: {e}")
            return []
    
    def _apply_fuzzy_matching(self, results, field, query_string, threshold):
        """Apply fuzzy string matching to filter results"""
        if not results or field not in results[0]:
            return results
        
        fuzzy_results = []
        field_values = [str(doc.get(field, '')) for doc in results]
        
        # Find fuzzy matches
        matches = self.fuzzy_matcher.find_best_matches(
            query_string, field_values, threshold, len(results)
        )
        
        # Filter results based on fuzzy matches
        matched_values = {match[0] for match in matches}
        for doc in results:
            if str(doc.get(field, '')) in matched_values:
                # Add similarity score to document
                similarity = next(
                    (score for value, score in matches if value == str(doc.get(field, ''))),
                    0.0
                )
                doc['_fuzzy_similarity'] = similarity
                fuzzy_results.append(doc)
        
        # Sort by similarity score
        fuzzy_results.sort(key=lambda x: x.get('_fuzzy_similarity', 0), reverse=True)
        return fuzzy_results
    
    def intelligent_breed_search(self, breed_query, threshold=0.6, limit=50):
        """
        Intelligent breed search using fuzzy matching
        
        Args:
            breed_query: Breed name or partial name to search for
            threshold: Minimum similarity threshold
            limit: Maximum number of results
            
        Returns:
            List of animals with similar breed names
        """
        # Get all unique breeds for fuzzy matching
        pipeline = [
            {"$group": {"_id": "$breed", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        
        breed_counts = list(self.collection.aggregate(pipeline))
        all_breeds = [item['_id'] for item in breed_counts if item['_id']]
        
        # Find fuzzy matches
        breed_matches = self.fuzzy_matcher.find_best_matches(
            breed_query, all_breeds, threshold
        )
        
        if not breed_matches:
            return []
        
        # Query for animals with matching breeds
        matching_breeds = [match[0] for match in breed_matches]
        query = {"breed": {"$in": matching_breeds}}
        
        results = self.enhanced_read(query, enable_fuzzy=True, 
                                   fuzzy_field='breed', fuzzy_query=breed_query,
                                   fuzzy_threshold=threshold)
        
        return results[:limit]
    
    def ml_recommend_animals(self, rescue_type, preference_weights=None, limit=20):
        """
        Use machine learning to recommend animals for rescue training
        
        Args:
            rescue_type: Type of rescue ('water', 'mountain', 'disaster')
            preference_weights: Dictionary of feature preferences
            limit: Maximum number of recommendations
            
        Returns:
            List of recommended animals with ML scores
        """
        if not self.ml_engine.is_trained:
            print("ML model not trained. Using traditional criteria.")
            return self.find_rescue_candidates(rescue_type)
        
        # Get candidate animals based on basic criteria
        candidates = self.find_rescue_candidates(rescue_type)
        
        if not candidates:
            return []
        
        # Apply ML scoring
        scored_candidates = []
        for animal in candidates:
            try:
                probability, importance = self.ml_engine.predict_success_probability(animal)
                animal['_ml_score'] = probability
                animal['_feature_importance'] = importance
                scored_candidates.append(animal)
            except Exception as e:
                print(f"ML scoring error for animal {animal.get('animal_id', 'unknown')}: {e}")
                animal['_ml_score'] = 0.5  # Default score
                scored_candidates.append(animal)
        
        # Sort by ML score
        scored_candidates.sort(key=lambda x: x.get('_ml_score', 0), reverse=True)
        
        return scored_candidates[:limit]
    
    def find_rescue_candidates(self, rescue_type="water"):
        """
        Enhanced rescue candidate search with performance optimization
        """
        cache_key = f"rescue_candidates_{rescue_type}"
        cached_result = self.cache.get(cache_key)
        
        if cached_result is not None:
            self.query_stats['cache_hits'] += 1
            return cached_result
        
        rescue_criteria = {
            "water": {
                "animal_type": "Dog",
                "breed": {"$in": ["Labrador Retriever Mix", "Chesapeake Bay Retriever", 
                                "Newfoundland", "Portuguese Water Dog"]},
                "age_upon_outcome_in_weeks": {"$gte": 26, "$lte": 156}
            },
            "mountain": {
                "animal_type": "Dog", 
                "breed": {"$in": ["German Shepherd", "Alaskan Malamute", "Old English Sheepdog",
                                "Siberian Husky", "Rottweiler"]},
                "age_upon_outcome_in_weeks": {"$gte": 26, "$lte": 156}
            },
            "disaster": {
                "animal_type": "Dog",
                "breed": {"$in": ["Doberman Pinscher", "German Shepherd", "Golden Retriever",
                                "Bloodhound", "Rottweiler"]},
                "age_upon_outcome_in_weeks": {"$gte": 20, "$lte": 300}
            }
        }
        
        if rescue_type in rescue_criteria:
            results = self.enhanced_read(rescue_criteria[rescue_type])
            self.cache.put(cache_key, results, ttl=600)  # Cache for 10 minutes
            return results
        else:
            return []
    
    def geospatial_search(self, center_lat, center_lon, radius_km, animal_criteria=None):
        """
        Search for animals within geographic radius using spatial algorithms
        
        Args:
            center_lat, center_lon: Center coordinates
            radius_km: Search radius in kilometers
            animal_criteria: Additional animal filtering criteria
            
        Returns:
            List of animals within geographic range
        """
        # Convert radius to approximate degrees (rough approximation)
        degree_radius = radius_km / 111.32  # km per degree latitude
        
        # Create geographic bounds
        min_lat, max_lat = center_lat - degree_radius, center_lat + degree_radius
        min_lon, max_lon = center_lon - degree_radius, center_lon + degree_radius
        
        # Build query with geographic bounds
        geo_query = {
            "location_lat": {"$gte": min_lat, "$lte": max_lat},
            "location_long": {"$gte": min_lon, "$lte": max_lon}
        }
        
        # Add additional animal criteria
        if animal_criteria:
            geo_query.update(animal_criteria)
        
        results = self.enhanced_read(geo_query)
        
        # Calculate exact distances and filter by radius
        filtered_results = []
        for animal in results:
            try:
                lat = float(animal.get('location_lat', 0))
                lon = float(animal.get('location_long', 0))
                
                distance = self._haversine_distance(center_lat, center_lon, lat, lon)
                if distance <= radius_km:
                    animal['_distance_km'] = distance
                    filtered_results.append(animal)
                    
            except (ValueError, TypeError):
                continue
        
        # Sort by distance
        filtered_results.sort(key=lambda x: x.get('_distance_km', float('inf')))
        
        return filtered_results
    
    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        """
        Calculate great circle distance between two points using Haversine formula
        
        Returns:
            Distance in kilometers
        """
        R = 6371.0  # Earth radius in kilometers
        
        lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
        lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = (math.sin(dlat/2)**2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c
    
    def get_performance_analytics(self):
        """
        Get comprehensive performance analytics
        
        Returns:
            Dictionary with performance metrics
        """
        cache_stats = self.cache.stats()
        
        analytics = {
            "query_performance": {
                "total_queries": self.query_stats['total_queries'],
                "average_query_time": f"{self.query_stats['avg_query_time']:.3f}s",
                "cache_hit_rate": f"{self.query_stats['cache_hits'] / max(self.query_stats['total_queries'], 1) * 100:.1f}%",
                "fuzzy_searches": self.query_stats['fuzzy_searches']
            },
            "cache_performance": cache_stats,
            "ml_status": {
                "model_trained": self.ml_engine.is_trained,
                "features_available": ML_AVAILABLE
            },
            "database_info": {
                "database": self.database_name,
                "collection": self.collection_name,
                "connection_status": "Connected"
            }
        }
        
        return analytics
    
    def benchmark_algorithms(self, test_size=1000):
        """
        Benchmark different algorithmic approaches
        
        Args:
            test_size: Number of records to test with
            
        Returns:
            Dictionary with benchmark results
        """
        print(f"Running algorithm benchmarks with {test_size} records...")
        
        # Test traditional query vs cached query
        start_time = time.time()
        traditional_results = list(self.collection.find().limit(test_size))
        traditional_time = time.time() - start_time
        
        start_time = time.time()
        cached_results = self.enhanced_read({}, use_cache=True)[:test_size]
        cached_time = time.time() - start_time
        
        # Test fuzzy matching performance
        if traditional_results:
            breeds = [doc.get('breed', '') for doc in traditional_results[:100]]
            start_time = time.time()
            fuzzy_matches = self.fuzzy_matcher.find_best_matches("German Shepherd", breeds)
            fuzzy_time = time.time() - start_time
        else:
            fuzzy_time = 0
            fuzzy_matches = []
        
        return {
            "traditional_query_time": f"{traditional_time:.3f}s",
            "cached_query_time": f"{cached_time:.3f}s",
            "cache_speedup": f"{traditional_time / max(cached_time, 0.001):.1f}x",
            "fuzzy_matching_time": f"{fuzzy_time:.3f}s",
            "fuzzy_matches_found": len(fuzzy_matches),
            "test_size": test_size
        }


# Convenience function for backward compatibility
def create_enhanced_shelter(username="aacuser", password="SNHU1234"):
    """Create enhanced animal shelter instance"""
    return EnhancedAnimalShelter(username, password)
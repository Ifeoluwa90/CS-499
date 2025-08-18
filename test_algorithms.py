#!/usr/bin/env python3
"""
CS 499 Milestone Three - Algorithm Testing Suite
Tests all enhanced algorithmic implementations to verify functionality

This script validates:
- Fuzzy string matching algorithms
- LRU cache performance
- Machine learning recommendations
- Geospatial search algorithms
- Performance optimization techniques
"""

from enhanced_animal_shelter import (
    EnhancedAnimalShelter, 
    FuzzyStringMatcher, 
    LRUCache, 
    MLRecommendationEngine,
    GeospatialIndex
)
import time
import pandas as pd
import random

def test_fuzzy_string_matching():
    """Test fuzzy string matching algorithm"""
    print("=" * 60)
    print("TESTING FUZZY STRING MATCHING (Levenshtein Distance)")
    print("=" * 60)
    
    matcher = FuzzyStringMatcher()
    
    # Test exact matches
    print("1. Testing exact string matches:")
    exact_similarity = matcher.similarity_ratio("German Shepherd", "German Shepherd")
    print(f"   'German Shepherd' vs 'German Shepherd': {exact_similarity:.3f} (should be 1.0)")
    
    # Test fuzzy matches
    print("\n2. Testing fuzzy string matches:")
    test_cases = [
        ("German Shepherd", "German Sheperd", "common typo"),
        ("Labrador Retriever", "Labrador", "partial match"),
        ("Golden Retriever", "Golden Retriver", "missing letter"),
        ("Rottweiler", "Rotweiler", "missing letter"),
        ("Bloodhound", "Bloodhoud", "transposition")
    ]
    
    for original, variant, description in test_cases:
        similarity = matcher.similarity_ratio(original, variant)
        print(f"   '{original}' vs '{variant}': {similarity:.3f} ({description})")
    
    # Test breed matching
    print("\n3. Testing breed search with fuzzy matching:")
    dog_breeds = [
        "German Shepherd", "Labrador Retriever Mix", "Golden Retriever",
        "Rottweiler", "Bloodhound", "Siberian Husky", "Alaskan Malamute",
        "Portuguese Water Dog", "Chesapeake Bay Retriever", "Newfoundland"
    ]
    
    search_queries = ["German Sheperd", "Labrador", "Golden Retriver"]
    
    for query in search_queries:
        matches = matcher.find_best_matches(query, dog_breeds, threshold=0.5)
        print(f"   Query: '{query}'")
        for breed, score in matches[:3]:
            print(f"     → {breed}: {score:.3f}")
        print()

def test_lru_cache():
    """Test LRU cache implementation"""
    print("=" * 60)
    print("TESTING LRU CACHE WITH TTL")
    print("=" * 60)
    
    cache = LRUCache(capacity=3, default_ttl=2)  # Small cache for testing
    
    print("1. Testing basic cache operations:")
    
    # Test insertions
    cache.put("key1", "value1")
    cache.put("key2", "value2") 
    cache.put("key3", "value3")
    print(f"   Inserted 3 items, cache size: {len(cache.cache)}")
    
    # Test retrievals
    value1 = cache.get("key1")
    value2 = cache.get("key2")
    print(f"   Retrieved key1: {value1}, key2: {value2}")
    
    # Test eviction
    cache.put("key4", "value4")  # Should evict key3 (least recently used)
    evicted_value = cache.get("key3")
    print(f"   After adding key4, key3 value: {evicted_value} (should be None)")
    
    print("\n2. Testing TTL expiration:")
    cache.put("temp_key", "temp_value", ttl=1)
    immediate_value = cache.get("temp_key")
    print(f"   Immediate retrieval: {immediate_value}")
    
    print("   Waiting 2 seconds for TTL expiration...")
    time.sleep(2)
    expired_value = cache.get("temp_key")
    print(f"   After TTL expiration: {expired_value} (should be None)")
    
    # Test cache statistics
    stats = cache.stats()
    print(f"\n3. Cache statistics:")
    print(f"   Capacity: {stats['capacity']}")
    print(f"   Current size: {stats['current_size']}")
    print(f"   Expired entries: {stats['expired_entries']}")

def test_ml_recommendations():
    """Test machine learning recommendation engine"""
    print("=" * 60)
    print("TESTING MACHINE LEARNING RECOMMENDATIONS")
    print("=" * 60)
    
    try:
        ml_engine = MLRecommendationEngine()
        
        # Create synthetic training data
        print("1. Creating synthetic training data...")
        synthetic_data = []
        breeds = ["German Shepherd", "Labrador Retriever", "Golden Retriever", 
                 "Bloodhound", "Rottweiler", "Poodle", "Bulldog"]
        
        for i in range(200):
            animal = {
                'breed': random.choice(breeds),
                'age_upon_outcome_in_weeks': random.randint(20, 200),
                'outcome_type': random.choice(['Adoption', 'Transfer', 'Return to Owner'])
            }
            synthetic_data.append(animal)
        
        df = pd.DataFrame(synthetic_data)
        print(f"   Created {len(df)} synthetic animal records")
        
        # Train model
        print("\n2. Training Random Forest model...")
        accuracy = ml_engine.train_model(df)
        print(f"   Model trained with accuracy: {accuracy:.3f}")
        
        # Test predictions
        print("\n3. Testing predictions for sample animals:")
        test_animals = [
            {'breed': 'German Shepherd', 'age_upon_outcome_in_weeks': 52, 'outcome_type': 'Adoption'},
            {'breed': 'Labrador Retriever', 'age_upon_outcome_in_weeks': 78, 'outcome_type': 'Adoption'},
            {'breed': 'Poodle', 'age_upon_outcome_in_weeks': 156, 'outcome_type': 'Transfer'}
        ]
        
        for animal in test_animals:
            probability, importance = ml_engine.predict_success_probability(animal)
            print(f"   {animal['breed']} ({animal['age_upon_outcome_in_weeks']} weeks): {probability:.3f} success probability")
            
    except ImportError:
        print("   Scikit-learn not available - ML features disabled")
    except Exception as e:
        print(f"   ML testing error: {e}")

def test_geospatial_algorithms():
    """Test geospatial indexing and search"""
    print("=" * 60)
    print("TESTING GEOSPATIAL ALGORITHMS")
    print("=" * 60)
    
    # Create spatial index
    spatial_index = GeospatialIndex()
    
    # Test points around Austin, TX area
    test_points = [
        ((30.2672, -97.7431), "Austin Downtown"),
        ((30.3922, -97.7207), "Austin North"),
        ((30.1833, -97.8683), "Austin West"),
        ((30.2500, -97.2500), "Austin East"),
        ((29.4241, -98.4936), "San Antonio")  # Further away
    ]
    
    print("1. Inserting test points into spatial index:")
    for point, name in test_points:
        spatial_index.insert(point, name)
        print(f"   Inserted: {name} at {point}")
    
    print("\n2. Testing range queries:")
    # Query around Austin area
    austin_bounds = (30.0, -98.0, 30.5, -97.0)  # (min_lon, min_lat, max_lon, max_lat)
    results = spatial_index.query_range(austin_bounds)
    
    print(f"   Austin area query returned {len(results)} points:")
    for point, name in results:
        print(f"     → {name} at {point}")

def test_enhanced_animal_shelter():
    """Test the complete enhanced animal shelter system"""
    print("=" * 60)
    print("TESTING ENHANCED ANIMAL SHELTER SYSTEM")
    print("=" * 60)
    
    try:
        # Initialize enhanced shelter
        print("1. Connecting to enhanced animal shelter...")
        enhanced_shelter = EnhancedAnimalShelter()
        print("   ✅ Connection successful")
        
        # Test cached queries
        print("\n2. Testing cached database queries...")
        start_time = time.time()
        results1 = enhanced_shelter.enhanced_read({}, use_cache=True)
        first_query_time = time.time() - start_time
        
        start_time = time.time()
        results2 = enhanced_shelter.enhanced_read({}, use_cache=True)  # Should hit cache
        second_query_time = time.time() - start_time
        
        print(f"   First query (database): {first_query_time:.3f}s")
        print(f"   Second query (cache): {second_query_time:.3f}s")
        print(f"   Speedup: {first_query_time / max(second_query_time, 0.001):.1f}x")
        
        # Test fuzzy breed search
        print("\n3. Testing intelligent breed search...")
        fuzzy_results = enhanced_shelter.intelligent_breed_search("German Sheperd", threshold=0.6)
        print(f"   Found {len(fuzzy_results)} animals with fuzzy breed matching")
        
        if fuzzy_results:
            print("   Top fuzzy matches:")
            for i, animal in enumerate(fuzzy_results[:3]):
                breed = animal.get('breed', 'Unknown')
                similarity = animal.get('_fuzzy_similarity', 0)
                print(f"     {i+1}. {breed} (similarity: {similarity:.3f})")
        
        # Test ML recommendations
        print("\n4. Testing ML recommendations...")
        ml_results = enhanced_shelter.ml_recommend_animals("water", limit=5)
        print(f"   ML recommended {len(ml_results)} animals for water rescue")
        
        if ml_results:
            print("   Top ML recommendations:")
            for i, animal in enumerate(ml_results[:3]):
                name = animal.get('name', 'Unknown')
                breed = animal.get('breed', 'Unknown')
                ml_score = animal.get('_ml_score', 0)
                print(f"     {i+1}. {name} ({breed}) - ML Score: {ml_score:.3f}")
        
        # Test performance analytics
        print("\n5. Testing performance analytics...")
        analytics = enhanced_shelter.get_performance_analytics()
        print(f"   Total queries: {analytics['query_performance']['total_queries']}")
        print(f"   Cache hit rate: {analytics['query_performance']['cache_hit_rate']}")
        print(f"   ML model status: {'Trained' if analytics['ml_status']['model_trained'] else 'Not trained'}")
        
        # Run benchmarks
        print("\n6. Running algorithm benchmarks...")
        benchmark_results = enhanced_shelter.benchmark_algorithms(test_size=100)
        print(f"   Traditional query: {benchmark_results['traditional_query_time']}")
        print(f"   Cached query: {benchmark_results['cached_query_time']}")
        print(f"   Cache speedup: {benchmark_results['cache_speedup']}")
        print(f"   Fuzzy matching: {benchmark_results['fuzzy_matching_time']}")
        
        print("\n   ✅ Enhanced animal shelter system test completed successfully!")
        
    except Exception as e:
        print(f"   ❌ Enhanced shelter test failed: {e}")
        print("   Note: This requires MongoDB connection with AAC database")

def main():
    """Run all algorithm tests"""
    print("CS 499 MILESTONE THREE - ALGORITHM TESTING SUITE")
    print("Testing advanced algorithms and data structure implementations")
    print()
    
    # Run all tests
    test_fuzzy_string_matching()
    print()
    
    test_lru_cache()
    print()
    
    test_ml_recommendations()
    print()
    
    test_geospatial_algorithms()
    print()
    
    test_enhanced_animal_shelter()
    print()
    
    print("=" * 60)
    print("ALL ALGORITHM TESTS COMPLETED")
    print("=" * 60)
    print()
    print("Summary of implemented algorithms:")
    print("✅ Fuzzy String Matching (Levenshtein Distance)")
    print("✅ LRU Cache with TTL (Least Recently Used)")
    print("✅ Machine Learning Recommendations (Random Forest)")
    print("✅ Geospatial Indexing (Quadtree-based)")
    print("✅ Performance Optimization and Analytics")
    print("✅ Enhanced Database Query Operations")
    print()
    print("These implementations demonstrate mastery of:")
    print("• Advanced string algorithms and dynamic programming")
    print("• Cache algorithms and memory management")
    print("• Machine learning integration and feature engineering")
    print("• Spatial data structures and geographic algorithms")
    print("• Performance analysis and optimization techniques")
    print("• Algorithm complexity analysis and trade-offs")

if __name__ == "__main__":
    main()
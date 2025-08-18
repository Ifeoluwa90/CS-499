// Enhanced Gaming Room Database Architecture
// CS 499 Milestone Four - Database Enhancement
// Author:Ifeoluwa Adewoyin
// This implementation demonstrates advanced database concepts including
// distributed systems, real-time data management, and cybersecurity principles

package com.gamingroom.enchanced;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CompletableFuture;
import java.sql.*;
import redis.clients.jedis.*;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.security.MessageDigest;
import javax.crypto.Cipher;
import javax.crypto.spec.SecretKeySpec;

/**
 * Enhanced Database Manager implementing distributed database architecture
 * for real-time gaming applications with security and performance optimization
 */
public class EnhancedGameDatabaseManager {
    
    // Database connection pools for different data types
    private final PostgreSQLConnectionPool postgresPool;
    private final JedisPool redisPool;
    private final InfluxDBClient influxClient;
    private final ElasticSearchClient elasticClient;
    
    // Security and encryption components
    private final SecurityManager securityManager;
    private final ObjectMapper jsonMapper;
    
    // Performance monitoring
    private final PerformanceMetrics metrics;
    
    /**
     * Constructor initializes all database connections and security components
     */
    public EnhancedGameDatabaseManager() {
        this.postgresPool = new PostgreSQLConnectionPool(
            "jdbc:postgresql://localhost:5432/gaming_room",
            "game_user", "secure_password"
        );
        this.redisPool = new JedisPool("localhost", 6379);
        this.influxClient = new InfluxDBClient("http://localhost:8086");
        this.elasticClient = new ElasticSearchClient("localhost", 9200);
        this.securityManager = new SecurityManager();
        this.jsonMapper = new ObjectMapper();
        this.metrics = new PerformanceMetrics();
        
        initializeDatabaseSchemas();
    }
    
    /**
     * Initialize PostgreSQL schemas for persistent game data
     */
    private void initializeDatabaseSchemas() {
        String[] schemas = {
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                username VARCHAR(50) UNIQUE NOT NULL,
                email VARCHAR(100) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                salt VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                total_games_played INTEGER DEFAULT 0,
                win_rate DECIMAL(5,2) DEFAULT 0.00,
                security_level INTEGER DEFAULT 1,
                failed_login_attempts INTEGER DEFAULT 0,
                account_locked_until TIMESTAMP NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS games (
                game_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                game_name VARCHAR(100) NOT NULL,
                created_by UUID REFERENCES users(user_id),
                game_status VARCHAR(20) CHECK (game_status IN ('waiting', 'active', 'completed', 'cancelled')),
                max_teams INTEGER DEFAULT 4,
                max_players_per_team INTEGER DEFAULT 4,
                rounds_total INTEGER DEFAULT 4,
                round_duration_seconds INTEGER DEFAULT 60,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                game_data_encrypted TEXT,
                integrity_hash VARCHAR(64)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS teams (
                team_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                game_id UUID REFERENCES games(game_id) ON DELETE CASCADE,
                team_name VARCHAR(50) NOT NULL,
                team_score INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                team_captain_id UUID REFERENCES users(user_id),
                UNIQUE(game_id, team_name)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS game_sessions (
                session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id UUID REFERENCES users(user_id),
                game_id UUID REFERENCES games(game_id),
                session_start TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                session_end TIMESTAMP,
                ip_address INET,
                user_agent TEXT,
                device_fingerprint VARCHAR(128),
                is_active BOOLEAN DEFAULT TRUE
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_games_status ON games(game_status);
            CREATE INDEX IF NOT EXISTS idx_games_created_by ON games(created_by);
            CREATE INDEX IF NOT EXISTS idx_teams_game_id ON teams(game_id);
            CREATE INDEX IF NOT EXISTS idx_sessions_user_game ON game_sessions(user_id, game_id);
            CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
            CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
            """
        };
        
        try (Connection conn = postgresPool.getConnection()) {
            for (String schema : schemas) {
                try (PreparedStatement stmt = conn.prepareStatement(schema)) {
                    stmt.execute();
                }
            }
        } catch (SQLException e) {
            throw new DatabaseInitializationException("Failed to initialize schemas", e);
        }
    }
    
    /**
     * Enhanced Game Management with Real-time Updates
     */
    public class EnhancedGameManager {
        
        /**
         * Create a new game with enhanced security and real-time capabilities
         */
        public CompletableFuture<GameInstance> createGame(
                String gameName, 
                UUID creatorId, 
                GameConfiguration config) {
            
            return CompletableFuture.supplyAsync(() -> {
                long startTime = System.currentTimeMillis();
                
                try {
                    // Validate input and check permissions
                    securityManager.validateGameCreation(creatorId, gameName);
                    
                    // Create persistent game record
                    UUID gameId = UUID.randomUUID();
                    String encryptedData = securityManager.encryptGameData(config);
                    String integrityHash = securityManager.calculateIntegrityHash(config);
                    
                    String sql = """
                        INSERT INTO games (game_id, game_name, created_by, game_status, 
                                         max_teams, max_players_per_team, rounds_total, 
                                         round_duration_seconds, game_data_encrypted, integrity_hash)
                        VALUES (?, ?, ?, 'waiting', ?, ?, ?, ?, ?, ?)
                        """;
                    
                    try (Connection conn = postgresPool.getConnection();
                         PreparedStatement stmt = conn.prepareStatement(sql)) {
                        
                        stmt.setObject(1, gameId);
                        stmt.setString(2, gameName);
                        stmt.setObject(3, creatorId);
                        stmt.setInt(4, config.maxTeams);
                        stmt.setInt(5, config.maxPlayersPerTeam);
                        stmt.setInt(6, config.roundsTotal);
                        stmt.setInt(7, config.roundDurationSeconds);
                        stmt.setString(8, encryptedData);
                        stmt.setString(9, integrityHash);
                        
                        stmt.executeUpdate();
                    }
                    
                    // Initialize real-time game state in Redis
                    try (Jedis redis = redisPool.getResource()) {
                        String gameStateKey = "game:" + gameId + ":state";
                        Map<String, String> initialState = new HashMap<>();
                        initialState.put("status", "waiting");
                        initialState.put("created_at", String.valueOf(System.currentTimeMillis()));
                        initialState.put("creator_id", creatorId.toString());
                        initialState.put("active_players", "0");
                        initialState.put("current_round", "0");
                        
                        redis.hmset(gameStateKey, initialState);
                        redis.expire(gameStateKey, 7200); // 2 hour expiration
                        
                        // Publish game creation event
                        GameEvent event = new GameEvent("GAME_CREATED", gameId, creatorId);
                        redis.publish("game_events", jsonMapper.writeValueAsString(event));
                    }
                    
                    // Log analytics event
                    recordAnalyticsEvent("game_created", gameId, creatorId, 
                                       Map.of("game_name", gameName, "config", config));
                    
                    // Record performance metrics
                    metrics.recordDatabaseOperation("create_game", 
                                                  System.currentTimeMillis() - startTime);
                    
                    return new GameInstance(gameId, gameName, creatorId, config);
                    
                } catch (Exception e) {
                    metrics.recordError("create_game", e);
                    throw new GameCreationException("Failed to create game: " + gameName, e);
                }
            });
        }
        
        /**
         * Real-time game state updates with conflict resolution
         */
        public CompletableFuture<Void> updateGameState(UUID gameId, GameStateUpdate update) {
            return CompletableFuture.runAsync(() -> {
                try (Jedis redis = redisPool.getResource()) {
                    String gameStateKey = "game:" + gameId + ":state";
                    String lockKey = "lock:" + gameStateKey;
                    
                    // Acquire distributed lock for atomic updates
                    String lockValue = UUID.randomUUID().toString();
                    if (!redis.set(lockKey, lockValue, "NX", "EX", 5).equals("OK")) {
                        throw new GameStateLockException("Unable to acquire lock for game: " + gameId);
                    }
                    
                    try {
                        // Get current state
                        Map<String, String> currentState = redis.hgetAll(gameStateKey);
                        
                        // Apply update with conflict resolution
                        Map<String, String> newState = resolveStateConflicts(currentState, update);
                        
                        // Validate state transition
                        securityManager.validateStateTransition(currentState, newState);
                        
                        // Update Redis state
                        redis.hmset(gameStateKey, newState);
                        redis.expire(gameStateKey, 7200);
                        
                        // Publish real-time update to subscribers
                        GameStateEvent stateEvent = new GameStateEvent(gameId, newState, update.getPlayerId());
                        redis.publish("game:" + gameId + ":updates", 
                                    jsonMapper.writeValueAsString(stateEvent));
                        
                        // Record analytics
                        recordAnalyticsEvent("game_state_update", gameId, update.getPlayerId(), newState);
                        
                    } finally {
                        // Release lock
                        String script = """
                            if redis.call('get', KEYS[1]) == ARGV[1] then
                                return redis.call('del', KEYS[1])
                            else
                                return 0
                            end
                            """;
                        redis.eval(script, Arrays.asList(lockKey), Arrays.asList(lockValue));
                    }
                }
            });
        }
        
        /**
         * Advanced game analytics with machine learning insights
         */
        public GameAnalytics generateGameAnalytics(UUID gameId, AnalyticsRequest request) {
            try {
                // Query InfluxDB for time-series data
                String influxQuery = String.format("""
                    SELECT 
                        mean(response_time) as avg_response_time,
                        count(*) as total_events,
                        sum(correct_guesses) as correct_guesses,
                        stddev(response_time) as response_time_variance
                    FROM game_events 
                    WHERE game_id = '%s' 
                    AND time >= %s AND time <= %s
                    GROUP BY team_id, round_number
                    """, gameId, request.getStartTime(), request.getEndTime());
                
                List<AnalyticsDataPoint> timeSeriesData = influxClient.query(influxQuery);
                
                // Query PostgreSQL for game metadata
                GameMetadata metadata;
                String sql = "SELECT * FROM games WHERE game_id = ?";
                try (Connection conn = postgresPool.getConnection();
                     PreparedStatement stmt = conn.prepareStatement(sql)) {
                    
                    stmt.setObject(1, gameId);
                    ResultSet rs = stmt.executeQuery();
                    
                    if (rs.next()) {
                        metadata = new GameMetadata(rs);
                    } else {
                        throw new GameNotFoundException("Game not found: " + gameId);
                    }
                }
                
                // Apply machine learning insights
                MLInsights insights = generateMLInsights(timeSeriesData, metadata);
                
                // Generate comprehensive analytics report
                return new GameAnalytics(gameId, timeSeriesData, metadata, insights);
                
            } catch (Exception e) {
                throw new AnalyticsGenerationException("Failed to generate analytics for game: " + gameId, e);
            }
        }
    }
    
    /**
     * Security Manager for database operations
     */
    public class SecurityManager {
        private final String ENCRYPTION_KEY = "YourSecretKey123"; // In production, use proper key management
        
        public void validateGameCreation(UUID userId, String gameName) {
            // Rate limiting check
            if (exceedsRateLimit(userId, "game_creation")) {
                throw new RateLimitExceededException("Game creation rate limit exceeded");
            }
            
            // Input validation
            if (gameName == null || gameName.trim().length() < 3 || gameName.length() > 100) {
                throw new InvalidInputException("Invalid game name");
            }
            
            // Check for malicious content
            if (containsMaliciousContent(gameName)) {
                throw new SecurityViolationException("Malicious content detected");
            }
        }
        
        public String encryptGameData(GameConfiguration config) {
            try {
                String json = jsonMapper.writeValueAsString(config);
                Cipher cipher = Cipher.getInstance("AES");
                SecretKeySpec keySpec = new SecretKeySpec(ENCRYPTION_KEY.getBytes(), "AES");
                cipher.init(Cipher.ENCRYPT_MODE, keySpec);
                byte[] encrypted = cipher.doFinal(json.getBytes());
                return Base64.getEncoder().encodeToString(encrypted);
            } catch (Exception e) {
                throw new EncryptionException("Failed to encrypt game data", e);
            }
        }
        
        public String calculateIntegrityHash(GameConfiguration config) {
            try {
                String json = jsonMapper.writeValueAsString(config);
                MessageDigest digest = MessageDigest.getInstance("SHA-256");
                byte[] hash = digest.digest(json.getBytes());
                return Base64.getEncoder().encodeToString(hash);
            } catch (Exception e) {
                throw new IntegrityException("Failed to calculate integrity hash", e);
            }
        }
        
        public void validateStateTransition(Map<String, String> currentState, Map<String, String> newState) {
            // Implement state transition validation logic
            String currentStatus = currentState.get("status");
            String newStatus = newState.get("status");
            
            // Define valid transitions
            Map<String, Set<String>> validTransitions = Map.of(
                "waiting", Set.of("active", "cancelled"),
                "active", Set.of("completed", "cancelled"),
                "completed", Set.of(),
                "cancelled", Set.of()
            );
            
            if (!validTransitions.get(currentStatus).contains(newStatus)) {
                throw new InvalidStateTransitionException(
                    String.format("Invalid transition from %s to %s", currentStatus, newStatus));
            }
        }
        
        private boolean exceedsRateLimit(UUID userId, String operation) {
            // Implementation for rate limiting check
            return false; // Simplified for example
        }
        
        private boolean containsMaliciousContent(String input) {
            // Implementation for malicious content detection
            String[] maliciousPatterns = {"<script", "javascript:", "eval(", "exec("};
            String lowerInput = input.toLowerCase();
            return Arrays.stream(maliciousPatterns).anyMatch(lowerInput::contains);
        }
    }
    
    /**
     * Performance monitoring and optimization
     */
    public class PerformanceMetrics {
        private final Map<String, List<Long>> operationTimes = new ConcurrentHashMap<>();
        private final Map<String, Integer> errorCounts = new ConcurrentHashMap<>();
        
        public void recordDatabaseOperation(String operation, long durationMs) {
            operationTimes.computeIfAbsent(operation, k -> new ArrayList<>()).add(durationMs);
            
            // Log slow operations
            if (durationMs > 1000) { // 1 second threshold
                System.err.println("Slow operation detected: " + operation + " took " + durationMs + "ms");
            }
        }
        
        public void recordError(String operation, Exception error) {
            errorCounts.merge(operation, 1, Integer::sum);
            System.err.println("Error in operation: " + operation + " - " + error.getMessage());
        }
        
        public Map<String, Double> getAverageOperationTimes() {
            Map<String, Double> averages = new HashMap<>();
            for (Map.Entry<String, List<Long>> entry : operationTimes.entrySet()) {
                double average = entry.getValue().stream().mapToLong(Long::longValue).average().orElse(0.0);
                averages.put(entry.getKey(), average);
            }
            return averages;
        }
    }
    
    /**
     * Analytics event recording for InfluxDB
     */
    private void recordAnalyticsEvent(String eventType, UUID gameId, UUID userId, Map<String, Object> data) {
        try {
            AnalyticsEvent event = new AnalyticsEvent(eventType, gameId, userId, data, System.currentTimeMillis());
            influxClient.writeEvent(event);
        } catch (Exception e) {
            System.err.println("Failed to record analytics event: " + e.getMessage());
        }
    }
    
    /**
     * Conflict resolution for concurrent game state updates
     */
    private Map<String, String> resolveStateConflicts(Map<String, String> current, GameStateUpdate update) {
        Map<String, String> resolved = new HashMap<>(current);
        
        // Apply vector clock or timestamp-based conflict resolution
        long currentTimestamp = Long.parseLong(current.getOrDefault("last_updated", "0"));
        long updateTimestamp = update.getTimestamp();
        
        if (updateTimestamp > currentTimestamp) {
            // Apply all updates from the newer timestamp
            resolved.putAll(update.getUpdates());
            resolved.put("last_updated", String.valueOf(updateTimestamp));
        } else {
            // Merge updates intelligently based on operation type
            for (Map.Entry<String, String> entry : update.getUpdates().entrySet()) {
                String key = entry.getKey();
                String newValue = entry.getValue();
                
                if (key.equals("score")) {
                    // For scores, use the higher value
                    int currentScore = Integer.parseInt(resolved.getOrDefault(key, "0"));
                    int newScore = Integer.parseInt(newValue);
                    resolved.put(key, String.valueOf(Math.max(currentScore, newScore)));
                } else {
                    // For other fields, use last-writer-wins
                    resolved.put(key, newValue);
                }
            }
        }
        
        return resolved;
    }
    
    /**
     * Machine Learning insights generation
     */
    private MLInsights generateMLInsights(List<AnalyticsDataPoint> data, GameMetadata metadata) {
        // Simplified ML insights - in production, use proper ML libraries
        Map<String, Object> insights = new HashMap<>();
        
        // Calculate performance patterns
        double avgResponseTime = data.stream()
            .mapToDouble(AnalyticsDataPoint::getResponseTime)
            .average()
            .orElse(0.0);
        
        // Identify optimal team sizes
        Map<Integer, Double> teamPerformance = data.stream()
            .collect(Collectors.groupingBy(
                AnalyticsDataPoint::getTeamSize,
                Collectors.averagingDouble(AnalyticsDataPoint::getSuccessRate)
            ));
        
        int optimalTeamSize = teamPerformance.entrySet().stream()
            .max(Map.Entry.comparingByValue())
            .map(Map.Entry::getKey)
            .orElse(4);
        
        insights.put("averageResponseTime", avgResponseTime);
        insights.put("optimalTeamSize", optimalTeamSize);
        insights.put("recommendedDifficulty", calculateRecommendedDifficulty(data));
        
        return new MLInsights(insights);
    }
    
    private String calculateRecommendedDifficulty(List<AnalyticsDataPoint> data) {
        double avgSuccessRate = data.stream()
            .mapToDouble(AnalyticsDataPoint::getSuccessRate)
            .average()
            .orElse(0.5);
        
        if (avgSuccessRate > 0.8) return "HARD";
        if (avgSuccessRate > 0.5) return "MEDIUM";
        return "EASY";
    }
    
    // Helper classes and interfaces
    public static class GameConfiguration {
        public int maxTeams = 4;
        public int maxPlayersPerTeam = 4;
        public int roundsTotal = 4;
        public int roundDurationSeconds = 60;
        public String difficulty = "MEDIUM";
        public boolean enableRealTimeUpdates = true;
        public boolean enableAnalytics = true;
    }
    
    public static class GameInstance {
        private final UUID gameId;
        private final String gameName;
        private final UUID creatorId;
        private final GameConfiguration config;
        
        public GameInstance(UUID gameId, String gameName, UUID creatorId, GameConfiguration config) {
            this.gameId = gameId;
            this.gameName = gameName;
            this.creatorId = creatorId;
            this.config = config;
        }
        
        // Getters
        public UUID getGameId() { return gameId; }
        public String getGameName() { return gameName; }
        public UUID getCreatorId() { return creatorId; }
        public GameConfiguration getConfig() { return config; }
    }
    
    // Additional helper classes would be defined here...
    public static class GameStateUpdate { 
        private final UUID playerId;
        private final Map<String, String> updates;
        private final long timestamp;
        
        public GameStateUpdate(UUID playerId, Map<String, String> updates) {
            this.playerId = playerId;
            this.updates = updates;
            this.timestamp = System.currentTimeMillis();
        }
        
        public UUID getPlayerId() { return playerId; }
        public Map<String, String> getUpdates() { return updates; }
        public long getTimestamp() { return timestamp; }
    }
    
    // Exception classes
    public static class DatabaseInitializationException extends RuntimeException {
        public DatabaseInitializationException(String message, Throwable cause) { super(message, cause); }
    }
    
    public static class GameCreationException extends RuntimeException {
        public GameCreationException(String message, Throwable cause) { super(message, cause); }
    }
    
    public static class RateLimitExceededException extends RuntimeException {
        public RateLimitExceededException(String message) { super(message); }
    }
    
    public static class InvalidInputException extends RuntimeException {
        public InvalidInputException(String message) { super(message); }
    }
    
    public static class SecurityViolationException extends RuntimeException {
        public SecurityViolationException(String message) { super(message); }
    }
    
    // I will be adding more classes soon
}

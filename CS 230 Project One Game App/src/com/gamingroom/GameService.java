package com.gamingroom;

import java.util.ArrayList;
import java.util.List;

/**
 * A singleton service for the game engine
 * 
 * @author Ifeoluwa Adewoyin
 */
public class GameService {

    /**
     * A list of the active games
     */
    private static List<Game> games = new ArrayList<Game>();

    /*
     * Holds the next game identifier
     */
    private static long nextGameId = 1;
    private static long nextPlayerId = 1;
    private static long nextTeamId = 1;

    /**
     * Singleton pattern implementation:
     * - Private static instance ensures only one instance exists
     * - Private constructor prevents external instantiation
     * - Public getInstance method provides global access to the single instance
     * This ensures that only one GameService object exists throughout the application,
     * maintaining a single point of control for game management.
     */
    private static GameService instance = null;

    private GameService() {
        // Private constructor to prevent instantiation
    }

    public static GameService getInstance() {
        if (instance == null) {
            instance = new GameService();
        }
        return instance;
    }

    /**
     * Construct a new game instance
     * 
     * @param name the unique name of the game
     * @return the game instance (new or existing)
     */
    public Game addGame(String name) {
        // Iterator to look for existing game with same name
        for (Game game : games) {
            if (game.getName().equals(name)) {
                return game;
            }
        }

        // If not found, make a new game instance and add to list of games
        Game game = new Game(nextGameId++, name);
        games.add(game);

        return game;
    }

    /**
     * Returns the game instance at the specified index.
     * <p>
     * Scope is package/local for testing purposes.
     * </p>
     * @param index index position in the list to return
     * @return requested game instance
     */
    Game getGame(int index) {
        return games.get(index);
    }
    
    /**
     * Returns the game instance with the specified id.
     * 
     * @param id unique identifier of game to search for
     * @return requested game instance
     */
    public Game getGame(long id) {
        // Iterator to look for existing game with same id
        for (Game game : games) {
            if (game.getId() == id) {
                return game;
            }
        }
        return null;
    }

    /**
     * Returns the game instance with the specified name.
     * 
     * @param name unique name of game to search for
     * @return requested game instance
     */
    public Game getGame(String name) {
        // Iterator to look for existing game with same name
        for (Game game : games) {
            if (game.getName().equals(name)) {
                return game;
            }
        }
        return null;
    }

    /**
     * Returns the number of games currently active
     * 
     * @return the number of games currently active
     */
    public int getGameCount() {
        return games.size();
    }

    /**
     * Returns the next player identifier
     * 
     * @return the next player identifier
     */
    public long getNextPlayerId() {
        return nextPlayerId++;
    }

    /**
     * Returns the next team identifier
     * 
     * @return the next team identifier
     */
    public long getNextTeamId() {
        return nextTeamId++;
    }
}

package com.gamingroom;

import java.util.ArrayList;
import java.util.List;

/**
 * A simple class to hold information about a team
 * <p>
 * Notice the overloaded constructor that requires
 * an id and name to be passed when creating.
 * Also note that no mutators (setters) defined so
 * these values cannot be changed once a team is
 * created.
 * </p>
 * @author Ifeoluwa Adewoyin
 *
 */
public class Team extends Entity {
    private List<Player> players = new ArrayList<Player>();

    /**
     * Constructor with an identifier and name
     */
    public Team(long id, String name) {
        super(id, name);
    }

    /**
     * Adds a player to the team
     * 
     * @param name the unique name of the player
     * @return the player instance (new or existing)
     */
    public Player addPlayer(String name) {
        // Check for existing player with same name
        for (Player player : players) {
            if (player.getName().equals(name)) {
                return player;
            }
        }

        // If not found, make a new player instance and add to list of players
        Player player = new Player(GameService.getInstance().getNextPlayerId(), name);
        players.add(player);
        return player;
    }

    @Override
    public String toString() {
        return "Team [id=" + id + ", name=" + name + ", players=" + players + "]";
    }
}

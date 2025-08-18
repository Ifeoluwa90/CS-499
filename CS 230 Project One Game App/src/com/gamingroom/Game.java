package com.gamingroom;

import java.util.ArrayList;
import java.util.List;

/**
 * A simple class to hold information about a game
 * 
 * <p>
 * Notice the overloaded constructor that requires
 * an id and name to be passed when creating.
 * Also note that no mutators (setters) defined so
 * these values cannot be changed once a game is
 * created.
 * </p>
 * 
 * @author Ifeoluwa Adewoyin
 *
 */
public class Game extends Entity {
    private List<Team> teams = new ArrayList<Team>();
    
    /**
     * Constructor with an identifier and name
     */
    public Game(long id, String name) {
        super(id, name);
    }

    /**
     * Adds a team to the game
     * 
     * @param name the unique name of the team
     * @return the team instance (new or existing)
     */
    public Team addTeam(String name) {
        // Check for existing team with same name
        for (Team team : teams) {
            if (team.getName().equals(name)) {
                return team;
            }
        }

        // If not found, make a new team instance and add to list of teams
        Team team = new Team(GameService.getInstance().getNextTeamId(), name);
        teams.add(team);
        return team;
    }

    @Override
    public String toString() {
        return "Game [id=" + id + ", name=" + name + ", teams=" + teams + "]";
    }
}

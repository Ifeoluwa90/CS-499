package com.gamingroom;

/**
 * A simple base class for Game, Team, and Player
 * 
 * @author Ifeoluwa Adewoyin
 */
public class Entity {
    protected long id;
    protected String name;

    /**
     * Constructor with an identifier and name
     */
    public Entity(long id, String name) {
        this.id = id;
        this.name = name;
    }

    /**
     * @return the id
     */
    public long getId() {
        return id;
    }

    /**
     * @return the name
     */
    public String getName() {
        return name;
    }

    @Override
    public String toString() {
        return "Entity [id=" + id + ", name=" + name + "]";
    }
}

package com.gamingroom;

/**
 * A class to test a singleton's behavior
 * 
 * @author Ifeoluwa Adewoyin
 */
public class SingletonTester {

    public void testSingleton() {
        
        System.out.println("\nAbout to test the singleton...");
        
        // FIXME: obtain local reference to the singleton instance
        GameService service = GameService.getInstance();
        
        // a simple test to ensure there is only one instance
        if (service == GameService.getInstance()) {
            System.out.println("The singleton works!");
        }
        else {
            System.out.println("We have a problem, the singleton failed!");
        }
    }
    
}

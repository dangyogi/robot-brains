# robot-brains

FTC "easy robot programming" alternative to Blockly (overly simple with no
support for modules) and Java (vastly too complicated with no real support for
robot programming).

# Goals

- First, and foremost, easy to program.

  - This is targeted at high school students who have never programmed before.

- Modularity

  - Allow the program to be represented as a set of independent bite-sized
    modules.
   
  - Allow re-use of modules across different programs (FTC "opmodes").

  - Support the creation of team libraries that can be re-used from season to
    season.
   
    - This allows teams to capture and reuse what they've learned, so that they
      can continually improve from one season to the next.

- Direct support for State Machines.

  - Often robot autonomous routines are diagrammed as state machines with
    chains of actions linked together, branching at decision points.
 
  - This language directly supports that using Decision Logic Tables for the
    branching decision points, and labeled blocks of code for actions that are
    linked with simple "goto" statements.

    - This language has no stack, so infinite loops of gotos are directly
      supported.

- Autonomous programming is easy.

  - With no stack, it is very easy to run multiple autonomous threads
    concurrantly.  For example, move forward 5 ft while raising the elevator
    14 inches.

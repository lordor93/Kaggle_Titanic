# Titanic: Machine Learning with Pipeline 
![1_Q59ONUgBl159KgMJvghShA](https://user-images.githubusercontent.com/13394756/127170079-6451ee3e-e18a-47c5-bd14-a30d00304f6f.jpeg)
# Objective
The goal is to predict the survival or the death of a given passenger based on a set of variables describing him such as his age, his sex, or his passenger class on the boat
# Data Dictionary
Variable	Definition	Key
survival - Survival	- 0 = No, 1 = Yes
pclass - Ticket class	-1 = 1st, 2 = 2nd, 3 = 3rd
sex	-     Sex	
Age	- Age in years	
sibsp - # of siblings / spouses aboard the Titanic	
parch	- # of parents / children aboard the Titanic	
ticket - Ticket number	
fare - 	Passenger fare	
cabin	- Cabin number	
embarked -	Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton
# Variable Notes
pclass: A proxy for socio-economic status (SES)
1st = Upper
2nd = Middle
3rd = Lower

age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5

sibsp: The dataset defines family relations in this way...
Sibling = brother, sister, stepbrother, stepsister
Spouse = husband, wife (mistresses and fiancés were ignored)

parch: The dataset defines family relations in this way...
Parent = mother, father
Child = daughter, son, stepdaughter, stepson
Some children travelled only with a nanny, therefore parch=0 for them.

# Note
To run the project, it is sufficient to run the "python main.py" command from the terminal in the project directory.

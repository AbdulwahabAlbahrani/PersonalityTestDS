# Personality prediction Model

  

## Abdulwahab Albahrani

  

---

  

## What is this project about?

  

---

  

## Aim Of the project

  

- Predicting the personality type of a person based on the a text sample of there writhing.

  

---

  

## MBTI

  

Stands for the Myers-Briggs Type Indicator

  

also known as the 16 personality test.

  

Personality test, that can be taken online for free, or a more in depth one that is paid.

  

---

  

Based on your answerers to some questions, you are given letter that represent aspects of your personality

  

---

  

### Indicators

  

- World

  

- Information

  

- Decisions

  

- Structure

  

---

### World

  

Do you prefer to focus on the outer world or on your own inner world?

  

Introverted(I) Or Extroverted(E)

  

---

  

### Information

  

Do you prefer to focus on the basic information you take in or do you prefer to interpret and add meaning?

  

Sensing(S) or Intuition(N)

  

---

  

### Decisions

  

When making decisions, do you prefer to first look at logic and consistency or first look at the people and special circumstances?

  

Thinking(T) or Feeling(F)

  

---

  

### Structure

  

In dealing with the outside world, do you prefer to get things decided or do you prefer to stay open to new information and options?

  

Judging(J) or Perceiving(P)

  

---

  

### Example

  

- INFP

  

- ESTJ

  

---

  

### Source

  

Zenodo.org

  

[Dataset](https://zenodo.org/record/1323873)

  

---

  

### Stages of the project

  

- Find the data set

  

- Import it.

  

- Explore it.

  

- Clean the data set.

  

- Train a neural network model.

  

- Tweak the model.

  

- Test the model.

  

---

  

### DataSet

  

This data set contains three columns:

  

- author_flair_text

  

- body

  

- subreddit

  

 and more than 1 mil and 600 entry were in the dataset.

---

because of the of the size of the dataset, I divided the dataset CSV file into 50 files to be able to be uploaded into GitHub.

---

![[Number split for indecators.png]]
(https://github.com/AbdulwahabAlbahrani/PersonalityTestDS/blob/master/Number of personality types (2).PNG)
---


![[Number of personality types (2).png]]
(https://github.com/AbdulwahabAlbahrani/PersonalityTestDS/blob/master/Number%20split%20for%20indecators.PNG)
---

### Data cleaning

- Removing the HTML tags in some of the posts.

- replacing links in posts with the word "URL"

- replacing the Emoticons and emojs with the appropriate word.

- Extracted from the author_flair_text the personality type and inserted it into a new column named "Type"

- removing the Nul values.

  

---

### Low accuracy score

---

### how is it fixed?

- I tried to change the parameters, the complicity of the neural network.

^^ Did not work

---

### The fix

- I found online that sometimes if you simplify the problem, it will be easer to solve

### Simplification

---

### Limitations:

**First**, because of time limitations, I was not able to get a model ready in time to show here.

  

---

### Improvements

- Try to change the model a bit to see if I can increase the accuracy of the model.

---

### Application

- this tool can be used for when approaching a new clint to know what is the best approach to approach them.

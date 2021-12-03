# Collaborative Filtering

**Collaborative filtering** is the task of making predictions about the interests of a user based on interests of many other users. As an example, let's look at the task of movie recommendation. Suppose we have 1,000,000 users, and a list of the movies each user has watched (from a catalog of 500,000 movies). Our goal is to recommend movies to users.

To solve this problem some method is needed to determine which movies are similar to each other. We can achieve this goal by embedding the movies into a low-dimensional space created such that similar movies are nearby.

![When moving to a two-dimensional movie embedding we now capture both how much the movie is geared towards children and also the degree to which it is a blockbuster or art-house film.](https://developers.google.com/machine-learning/crash-course/images/Embedding2dWithLabels.svg?dcb_=0.9146034342940375)

With this two-dimensional embedding we define a distance between movies such that movies are nearby (and thus inferred to be similar) if they are both alike in the extent to which they are geared towards children versus adults, as well as the extent to which they are blockbuster movies versus arthouse movies. These, of course, are just two of many characteristics of movies that might be important.

More generally, what we've done is mapped these movies into an **embedding space**, where each word is described by a two-dimensional set of coordinates. For example, in this space, "Shrek" maps to (-1.0, 0.95) and "Bleu" maps to (0.65, -0.2). In general, when learning a *d*-dimensional embedding, each movie is represented by *d* real-valued numbers, each one giving the coordinate in one dimension.

In this example, we have given a name to each dimension. When learning embeddings, the individual dimensions are not learned with names. Sometimes, we can look at the embeddings and assign semantic meanings to the dimensions, and other times we cannot. Often, each such dimension is called a **latent dimension**, as it represents a feature that is not explicit in the data but rather inferred from it.

Ultimately, it is the distances between movies in the embedding space that are meaningful, rather than a single movie's values along any given dimension.
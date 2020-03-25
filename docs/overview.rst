Categorical Structure
=====================

What do we mean by the *structure* of a categorical variable?

The standard approaches to dealing with categorical variables are "one-hot encoding" (which assumes the possible values are all different from one another, but with no particular similarities or differences) or "integer-encoding" (which assumes the values have an ordinal or linear structure).

However, in many cases the underlying structure of the values is not a strict ordering.

Here are a few examples:

- The contiguous 48 states in the USA have a structure based on which states border which others.
- The 12 months in the year have a circular structure: January is "next to" December in the same way that July is "next to" August.
- A survey question has a set of possible response given by "Poor", "Fair", "Good", "Excellent", "Not Applicable", "I prefer not to answer".  The first four answers may have a linear structure, with the other two responses incomparable.
- The outcome variable of CIFAR-10 has the following possibilities: airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks. 5 of them are vehicles and 5 are animals.  Of the 5 animals, 3 are mammals.  This yields an ontological structure: a car is more similar to a truck than either is to a frog.

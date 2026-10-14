

# Dataset

Ok chat, looks like things are working. Now let's scrutinize the dataset. I want to make sure we're on the same page. We're building a model here for LPNs to work with Pattern-2D. My understanding is that you built a generator class/function here for making such Pattern-2D inputs/outputs. Is that correct? I want to fully understand the dimensions of this. Are the inputs 4x4 grids with a blue pixel, followed by outputs that are still 4x4 grids, but with a patterned tile placed at the top-left where that pixel was? Are these inputs/outputs constrained by where the colored tile can be placed (e.g., rows 0-2 and columns 0-2)? 

# Architecture

Great. Now let's analyze the architecture. Describe the architecture we've implemented. We are using a Pattern-2D checkpoint model, correct? Describe the MLP we've implemented in particular: its details, its purpose, etc.


# Experiments and Smoke Tests

Ok, now to the experiments. The only experiment we're really implementing is matching accuracy, right? 


# Tables and Visualizations

# Animations of Programs

# Architecture Print-Out


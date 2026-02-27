# Notebooks

1. Use `JupyterTOC` with `Minimum Header Level = 2` (see Settings)
1. Notebook tile is `# TITLE`
1. Section are `##`, `###`, ...
1. All imports in initial `# Import` section
1. Most settings in initial `# Settings` section
1. No spaces in filename

#  Imports: Most general first

Good:

    import os
    import numpy

    import localmodule

Bad:

    import localmodule
    import numpy
    import os

# Matplotlib

Import as 
    
    ```
    import matplotlib as plt.
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plt.rcParams.update({'axes.grid':True,'grid.color':'black','grid.alpha':'0.25','grid.linestyle':'--'})
    plt.rcParams.update({'font.size':14})
    plt.rcParams.update({'font.family':'serif'})
    ```

1. Use `fig,ax = plt.subplots()` (or `fig = plt.figure(); ax = fig.add_subplot`)
1. Use `figsize = (6,4)` as default
1. Labels are lowercase

# Function calls and indexing: No space after commas 

`x = f(a,b)`, not `x = f(a, b)`
`x = y[i,j]`, not `x = y[i, j]`

# Comments

1. Ordered, a., b., c., then i., ii., iii., then o., oo., oo.
1. Keep to a minimum (assume basic code knowledge)
1. Lowercase: 'a. compute',  not 'a. Compute'
1. Directly after line (not tab aligned or similar)

    x = 1 # good
    y = 2       # bad 1 
    abc_abc = 4 # bad 2

1. Only very few commented out code lines
1. Blank line above

Good:

    # a. initialize x
    x = 0

    # b. initialy y
    y = 1

Bad:

    # a. initialize x
    x = 0
    # b. initialy y
    y = 1

# Math: x = a*x**2 + y/5

1. Spaces around =, + and -
1. No spaces around *, /, and **
1. No tab alignment like

    a   = 1
    abc = 2

# Strings

1. Use ' rather than "
1. Use f'{x}_{y}' rather x + '_' + y
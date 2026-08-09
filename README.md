# MathOptimization

Last update: 8/9/2026.

I am currently writing an optimization textbook and this library will host all of the algorithms in it. I am expecting the book to be done be EoQ4 '26/BoQ1 '27.

The main file uses three solvers which use different methodologies** to come to solutions.

It can solve convex problems globally
It can solve nonconvex problems globally (functions must be at least Lipschitz - your functions probably are)
By convention, all optimizations are minimization.
I believe my maximin_solver is an original algorithm, if you know otherwise, let me know.

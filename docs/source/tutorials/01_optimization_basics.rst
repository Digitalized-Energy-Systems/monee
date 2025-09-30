======================
Optimization basics
======================

To formulate an optimization problem, there are helper classes to make that as effortless as possible. In the following we want the optimizer to shedd as much loads as necessary (simplified).

.. testcode::

    import monee.problem as mp
    import monee.model as mm

    # create problem (debug=true -> more logging on that problem instance)
    problem = mp.OptimizationProblem(debug=True)

    # define which variables are controllable by the solver
    # here we mark all 'regulation' fields of the demands as controllable
    # the AttributeParameter instance defines the min/max and initial value
    problem.controllable_demands((
        "regulation",
        mp.AttributeParameter(
            min=lambda attr, val: 0, max=lambda attr, val: 1, val=lambda attr, val: 1
        ),
    ))
    # with bounds we define bounds on selected models/attributes
    problem.bounds((0.9,1.1), lambda m, _: type(m) is mm.Bus, ["vm_pu"])

    # to calculate the objective we can use the controllables_link, which allows us to specifiy
    # a lambda which assigns data to each controllable. After that we can call calculate which expects
    # a lambda with one argument 'model_to_data'
    objectives = mp.Objectives()
    objectives.with_models(problem.controllables_link).data(
        lambda model: 10
    ).calculate(lambda model_to_data: sum(model_to_data.values()))

    # Further we could specify custom constraints like with selecting model types and then
    # specifying the constraint equation on these set of models
    constraints = mp.Constraints()
    constraints.select_types(mm.ExtPowerGrid).equation(
        lambda model: model.p_mw >= 0
    )

    # at the end we just assign the objectives and constrains container an the problem is ready
    problem.constraints = constraints
    problem.objectives = objectives


To use an OptimizationProblem, you can pass it to :meth:`monee.run_energy_flow_optimization`

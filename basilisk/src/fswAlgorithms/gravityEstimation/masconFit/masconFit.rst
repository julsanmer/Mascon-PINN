Executive Summary
-----------------
This module fits a mascon distribution using position-gravity acceleration data. The resulting mascon distribution is ensured to have positive masses contained within the gravity body shape.

This module encodes a gradient descent algorithm to minimize a loss function. Consequently, an initial mascon distribution needs to be set.

Message Connection Descriptions
-------------------------------
This module has no message connections and its main interface is called as a method.


Detailed Module Description
---------------------------
General Function
^^^^^^^^^^^^^^^^
The ``masconFit()`` module receives position-acceleration data (:math:`m` samples)

.. math::
    :label: eq:data

    \mathbf{r}_{\textup{data}} =
    \begin{bmatrix}
    \mathbf{r}_1 \\
    \vdots \\
    \mathbf{r}_m\\
    \end{bmatrix}, \>\>\>\>\>\>\>\>
    \mathbf{a}_{\textup{data}} =
    \begin{bmatrix}
    \mathbf{a}_1 \\
    \vdots \\
    \mathbf{a}_m \\
    \end{bmatrix},

in order to fit a mascon distribution of :math:`n+1` point masses. Each point mass is characterized by its standard gravity and its position. The mascon model gravity acceleration is

.. math::
    :label: eq:

    \mathbf{a}_M=-\sum^{n}_{k=0}\mu_{M_k}\frac{\mathbf{r}-\mathbf{r}_{M_k}}{\rVert\mathbf{r}-\mathbf{r}_{M_k}\lVert^3},

thus the mascon distribution is characterized by :math:`\mu_{M_k}` and :math:`\mathbf{r}_{M_k}`.

The associated frame definitions may be found in the following table. Since only the gravity body centred rotating frame is used, its sub-index is ommited from now on.

.. list-table:: Frame Definitions
    :widths: 25 25
    :header-rows: 1

    * - Frame Description
      - Frame Definition
    * - Gravity body centred rotating frame
      - :math:`A: \{\hat{\mathbf{a}}_1, \hat{\mathbf{a}}_2, \hat{\mathbf{a}}_3\}`

Preprocessing
^^^^^^^^^^^^^^

Each time the main method ``train(posData, accData, show_progress)`` is called, the 0th mass is expressed as :math:`\mu_0=\mu-\sum^n_{k=1}\mu_k` (which ensures the total mass remains constant during training) and its position is locked throughout the training. This changes the acceleration dataset by adding a constant term.
 

Algorithm
^^^^^^^^^^
This module uses Adam gradient descent to minimize a loss function based on the mismatch between mascon prediction and data. The loss is defined to minimize the gravity mean squared error as

.. math::
    :label: eq:loss

    L=\frac{1}{m}\sum^{m}_{j=1}\frac{\rVert\mathbf{a}_M(\mathbf{r}_j)-\mathbf{a}_{j}\lVert^2}{\lVert\mathbf{a}_j\rVert^2}

Alternatively, the user can also choose to minimize the mean linear error.

In order to minimize the loss, the Adam algorithm is used. Adam performs the following ith step until a maximum number of iterations is reached:

1) Compute loss gradient:

.. math::
    :label: eq:loss_gradient

    \nabla L^{[i-1]}\leftarrow\nabla L(\mathbf{y}^{[i-1]})

2) Update biased moments:

.. math::
    :label: eq:bias_moments

    \mathbf{m}^{[i]}\leftarrow\beta_1\mathbf{m}^{[i-1]}+(1-\beta_1)\nabla L^{[i-1]}\\
    \mathbf{v}^{[i]}\leftarrow\beta_2\mathbf{v}^{[i-1]}+(1-\beta_2)(\nabla L^{[i-1]})^2

3) Correct bias:

.. math::
    :label: eq:unbiased_moments

    \hat{\mathbf{m}}^{[i]}\leftarrow\mathbf{m}^{[i]}/(1-\beta^i_1)\\
    \hat{\mathbf{v}}^{[i]}\leftarrow\mathbf{v}^{[i]}/(1-\beta^i_2)

4) Update decision variable:

.. math::
    :label: eq:update

    \mathbf{y}^{[i]}\leftarrow\mathbf{y}^{[i-1]}-\eta\hat{\mathbf{m}}^{[i]}/(\sqrt{\hat{\mathbf{v}}^{[i]}}+\epsilon)

5) Do the constraints projection step:

.. math::
    :label: eq:constraints

    \mathbf{y}^{[i]}\leftarrow\mathbf{g}(\mathbf{y}^{[i]})


The last step ensures the masses positiveness and that they are interior to the body shape. More details on the internal algorithm implementation can be found in `Sanchez and Schaub <https://doi.org/10.48550/arXiv.2305.07333>`__ (see section IV and Appendix A).


Module Assumptions and Limitations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The module assumptions and limitations are listed below:

 - The reference frame is the body centred rotating frame.
 - The total standard gravity :math:`\mu` has to be set and is kept constant during training.
 - The first mass position is locked and cannot be varied during training.
 - The interior constraint (when mascon positions are trained) is based on a polyhedron shape.
 - The gradient descent algorithm is Adam.


User Guide
----------

To use this module, instantiate the class and provide it with the necessary parameters (more details below). The main method is ``masconfit.train(pos_data, acc_data, show_progress)``. The inputs ``pos_data`` and ``acc_data`` are :math:`N\times3` matrices. The last argument ``show_progress`` is a boolean. No message connections are required.

A training based on the masconFit module can be created by following these instructions. First, import necessary packages:

.. code-block:: python

    # Import packages
    import numpy as np
    from Basilisk.fswAlgorithms import masconFit
    from Basilisk.simulation.gravityEffector import loadPolyFromFileToList
    from Basilisk.utilities import simIncludeGravBody
    from Basilisk import __path__

    bsk_path = __path__[0]

Then, create a gravity body and generate a position-acceleration dataset

.. code-block:: python

    # Create a gravity body to generate gravity acceleration data
    mu = 4.27 * 1e5 # Total standard gravity
    gravFactory = simIncludeGravBody.gravBodyFactory()
    gravity = gravFactory.createCustomGravObject("asteroid", mu=mu)
    simIncludeGravBody.loadPolyFromFile(polyFile, gravity.poly)
    gravity.poly.initializeParameters()

    # Create dataset
    nData = 100 # Number of samples
    r_min = 18 * 1e3 # Minimum data radius
    r_max = 30 * 1e3 # Maximum data radius
    pos_data = np.zeros((nData, 3)) # Position data
    acc_data = np.zeros((nData, 3)) # Acceleration data
    np.random.seed(0)
    for i in range(nData):
        # Generate random radius, longitude and latitude
        r = np.random.uniform(r_min, r_max)
        lon = np.random.uniform(0, 2*np.pi)
        lat = np.random.uniform(-np.pi/2, np.pi/2)

        # Add position-acceleration sample
        pos_data[i, 0:3] = r*np.array([np.cos(lon)*np.cos(lat),
                                       np.sin(lon)*np.cos(lat),
                                       np.sin(lat)])
        acc_data[i, 0:3] = np.array(gravity.poly.computeField(pos_data[i,0:3].tolist())).reshape(3)

Now, instantiate the module class, set a mascon distribution, mascon adimensional factors and a polyhedron shape to account for the interior constraint:

.. code-block:: python

    # Instantiate module class
    masconfit = masconFit.MasconFit()
    
    # Define an initial mascon distribution (this is just an example thus few masses are set)
    muM0 = np.array([0.2, 0.2, 0.2, 0.2, 0.2]) * mu # Masses standard gravity
    xyzM0 = np.array([[0, 0, 0], [10, 0, 0], [-10, 0, 0],
                    [0, 5, 0], [0, -5, 0]])*1e3 # Masses position
    nM = len(muM0) # Number of masses

    # Set the initial mascon distribution in the class
    masconfit.mu = mu
    masconfit.muM = muM0.tolist()
    masconfit.xyzM = xyzM0.tolist()

    # Set adimensional parameters (helps the training)
    masconfit.muMad = mu / nM # Mass adimensionalization (optional)
    masconfit.xyzMad = [1.6*1e3, 0.8*1e3, 0.4*1e3] # Positions adimensionalization (optional)

    # Set training variables flag
    masconfit.trainXYZ = True # True if mascon masses-positions are to be fitted; False if only mascon masses are to be fitted (optional)

    # Initialize a polyhedron shape (only required if trainXYZ = True)
    polyFile = bsk_path + '/supportData/LocalGravData/EROS856Vert1708Fac.txt'
    xyz_vert, order_face, n_vert, n_face = loadPolyFromFileToList(polyFile)
    masconfit.shape.initPolyhedron(xyz_vert, order_face)

Finally, set loss type, Adam hyperparameters and execute the training of the previously generated dataset:

.. code-block:: python

    # Set loss function type
    masconfit.useMSE = True # Flag for gravity mean-squared error to be minimized (optional)
    masconfit.useMLE = False # Flag for gravity mean-linear error to be minimized (optional)

    # Set Adam gradient descent parameters
    masconfit.setMaxIter(1000) # Sets maximum number of iterations (optional)
    masconfit.setLR(1e-3) # Sets learning rate (optinal)
    masconfit.setHyperparam(0.9, 0.99, 1e-6) # Sets beta1, beta2 and eps hyperparameters (optional)
    
    # Execute the fitting process
    show_progress = True # When True, iterations and loss evolution are printed
    masconfit.train(pos_data.tolist(), acc_data.tolist(), show_progress) # Call to main method
    
    # Get the fitted mascon distribution
    muM = np.array(masconfit.muM)
    xyzM = np.array(masconfit.xyzM)
    
    # Get the loss evolution
    loss = np.array(masconfit.getLoss()).squeeze()

To resume, the user must set the following variables

- ``mu``, total gravitational constant in :math:`\text{m}^3/\text{s}^2`
- ``muM``, initial mascon standard gravity vector in :math:`\text{m}^3/\text{s}^2`
- ``xyzM``, initial mascon position matrix in :math:`\text{m}`

The user could opt to set the following module variables (initialized by default):

- ``muMad``, mascon standard gravity adimensionalization factor (1 by default)
- ``xyzMad``, mascon position adimensionalization factor ([1,1,1] by default)
- ``useMSE``, mean-squared error loss flag (``true`` by default)
- ``useMLE``, mean-linear error loss flag (``false`` by default)
- ``trainXYZ``, flag to fit mascon positions (``false`` by default)
- ``graddescent->lr`` learning rate (:math:`10^{-3}` by default)
- ``graddescent->maxiter`` maximum iterations (:math:`1000` by default)
- ``graddescent->beta1`` average gradient decay factor (:math:`0.9` by default)
- ``graddescent->beta2`` average squared gradient decay factor (:math:`0.99` by default)
- ``graddescent->eps`` numerical stability constant (:math:`10^{-6}` by default)

If ``trainXYZ=true`` is set, the user must provide a polyhedron shape by using the method ``shape.initPolyhedron(xyz_vert, order_face)`` where ``xyz_vert`` are the vertexes position and ``order_face`` the vertex indexes of each face.

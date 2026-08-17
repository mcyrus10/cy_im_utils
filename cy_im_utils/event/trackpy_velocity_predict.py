#%% [markdown]
# # quantized velocity predictor for trackpy

#%%
import trackpy.predict
import numpy as np

def predict_handle(U, V, grid_size, default_x):
    """
    curried function to pass a quantized velocity field into trackpy to improve
    the linking

    usage:
        U,V = ....<use a function to calculate the velcotiy field>
        predictor = predict_handle(U ,V, grid_size, 0)
        tracks = tp.link(<dataframe>, search_range = <range>, memory = <mem>, predictor = predictor)
    """
    @trackpy.predict.predictor
    def predict(t1, particle):
        # This is supposed to be close to the velocity in fakeframe().
        # Note that the default order for coordinates in trackpy is (y, x).
        # You will rarely have to know this unless you are doing something
        # nerdy like making your own predictor.
        grid_y, grid_x = np.round((np.array(particle.pos) // grid_size)).astype(int)
        spatial_args = (grid_y, grid_x)
        v_y = V[spatial_args]
        v_x = U[spatial_args]
        if (v_x == 0 and v_y == 0):
            velocity = np.array([0, default_x])
        else:
            velocity = np.array([v_y, v_x])  # (v_y, v_x)
        return particle.pos + velocity * (t1 - particle.t)

    return predict

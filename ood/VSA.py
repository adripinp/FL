import tensorflow as tf

class Vsa:
    """ VSA/HDC mathematics form the backbone of the ood detection approach.
    References:
        - https://arxiv.org/abs/2112.05341
        - https://github.com/SamWilso/HDFF_Official
        
        However code is re-written to fit tensorflow and this project structure.
    """
    debug = False
    
    def __init__(self, debug) -> None:
        self.debug = debug
    
    def _dim_check(self, x, y):
        if len(tf.shape(x)) < 2:
            x = tf.expand_dims(x, axis=0)
        if len(tf.shape(y)) < 2:
            y = tf.expand_dims(y, axis=0)
        return x, y

    def bundle(self, x, y) -> tf.Tensor:
        x, y = self._dim_check(x, y)
        return x + y
	
    def bulk_bundle(self, x) -> tf.Tensor:
        return tf.reduce_sum(x, axis=0)
    
    def bind(self, x, y) -> tf.Tensor:
        x, y = self._dim_check(x, y)
        return x * y

    def norm(self, tensor):
        return tf.nn.l2_normalize(tensor, axis=1)  # Normalize along feature axis

    def similarity(self, x, y):
        # Both should be (n_samples, hyper_dim)
        x, y = self._dim_check(x, y)
        x, y = self.norm(x), self.norm(y)
        return tf.linalg.matmul(x, y, transpose_b=True)
    
    def euclidean_distance(self, x, y):
        """
        Compute the pairwise Euclidean distance between two tensors.

        Args:
            x (tf.Tensor): Tensor of shape (n_samples, n_features).
            y (tf.Tensor): Tensor of shape (m_samples, n_features).

        Returns:
            tf.Tensor: Pairwise Euclidean distances of shape (n_samples, m_samples).
        """
        x, y = self._dim_check(x, y)
        # Compute squared differences
        squared_diff = tf.expand_dims(x, axis=1) - tf.expand_dims(y, axis=0)
        squared_distances = tf.reduce_sum(tf.square(squared_diff), axis=-1)
        distances = tf.sqrt(squared_distances)

        return distances

if __name__ == "__main__": 
    hdff = Vsa(debug=True)

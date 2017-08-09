import mdp.nodes
import mdp.utils
import mdp.parallel
import scipy.sparse as sp
from datetime import datetime
import logging


# Ridge Regression node
class RidgeRegressionNode(mdp.nodes.LinearRegressionNode):
    """Ridge Regression node. Extends the LinearRegressionNode and adds an additional
    ridge_param parameter.
    
    Solves the following equation: (AA^T+\lambdaI)^-(1)A^TB
    
    It is also possible to define an equivalent noise variance: the ridge parameter
    is set such that a regularization equal to a given added noise variance is
    achieved. Note that setting the ridge_param has precedence to the eq_noise_var.
    """

    # Constructor
    def __init__(self, ridge_param=0, eq_noise_var=0, with_bias=True, use_pinv=False, input_dim=None, output_dim=None, dtype=None):
        super(RidgeRegressionNode, self).__init__(input_dim=input_dim, output_dim=output_dim, with_bias=with_bias, use_pinv=use_pinv, dtype=dtype)

        self.ridge_param = ridge_param
        self.eq_noise_var = eq_noise_var
        self.with_bias = with_bias

    # Train the node
    def _train(self, x, y):
        """
        **Additional input arguments**

        y
          array of size (x.shape[0], output_dim) that contains the observed
          output to the input x's.
        """
        # initialize internal vars if necessary
        logging.getLogger(name=u"Oger").info(u"Initialize internal vars if necessary")
        if self._xTx is None:
            if self.with_bias:
                x_size = self._input_dim + 1
            else:
                x_size = self._input_dim
            # end if
            self._xTx = mdp.numx.zeros((x_size, x_size), self._dtype)
            self._xTy = mdp.numx.zeros((x_size, self._output_dim), self._dtype)
        # end if

        # Add bias
        if self.with_bias:
            x = self._add_constant(x)
        # end if
        
        # Compute x^T * x (x = reservoir's states)
        logging.getLogger(name=u"Oger").info(u"Compute x^T * x (xTx)")
        self._xTx += mdp.utils.mult(x.T, x)

        # Compute x^T * y (y = target outputs)
        # Y is a sparse matrix or not?
        logging.getLogger(name=u"Oger").info(u"Compute x^T * y (xTy)")
        if type(y) is sp.csr_matrix:
            self._xTy += x.T * y
        else:
            self._xTy += mdp.utils.mult(x.T, y)
        # end if

        self._tlen += x.shape[0]
    # end _train

    # Stop the training and compute W_output
    def _stop_training(self):
        """
        Stop the training and compute W_output
        """
        try:
            if self.use_pinv:
                invfun = mdp.utils.pinv
            else:
                invfun = mdp.utils.inv
            # end if

            if self.ridge_param > 0:
                lmda = self.ridge_param
            else:
                lmda = self.eq_noise_var ** 2 * self._tlen
            # end if

            # Inverse matrix xTx
            logging.getLogger(name=u"Oger").info(u"Inverse matrix xTx")
            inv_xTx = invfun(self._xTx + lmda * mdp.numx.eye(self._input_dim + self.with_bias))
        except mdp.numx_linalg.LinAlgError, exception:
            errstr = (str(exception) +
                      "\n Input data may be redundant (i.e., some of the " +
                      "variables may be linearly dependent).")
            raise mdp.NodeException(errstr)
        # end try

        # Compute W_output
        logging.getLogger(name=u"Oger").info(u"Compute W_output")
        self.beta = mdp.utils.mult(inv_xTx, self._xTy)
        # self.beta = mdp.numx.linalg.solve(self._xTx + lmda * mdp.numx.eye(self._input_dim + 1), self._xTy)
    # end _stop_training
        
    def _execute(self, x):
        """
        Execute the node
        :param x:
        :return:
        """
        if self.with_bias:
            x = self._add_constant(x)
        output = mdp.utils.mult(x, self.beta)
        return output
    # end _execute

# end RidgeRegressionNode

class ParallelLinearRegressionNode(mdp.parallel.ParallelExtensionNode, mdp.nodes.LinearRegressionNode):
    """Parallel extension for the LinearRegressionNode and all its derived classes
    (eg. RidgeRegressionNode).
    """
    def _fork(self):
        return self._default_fork()

    def _join(self, forked_node):
        if self._xTx is None:
            self._xTx = forked_node._xTx
            self._xTy = forked_node._xTy
            self._tlen = forked_node._tlen
        else:
            self._xTx += forked_node._xTx
            self._xTy += forked_node._xTy
            self._tlen += forked_node._tlen

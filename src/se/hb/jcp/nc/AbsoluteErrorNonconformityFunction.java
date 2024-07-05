package se.hb.jcp.nc;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.AbstractMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import se.hb.jcp.bindings.deeplearning4j.NN4jRegressor;
import se.hb.jcp.bindings.smile.GaussianProcessRegressor;
import se.hb.jcp.ml.IRegressor;

public class AbsoluteErrorNonconformityFunction implements IRegressionNonconformityFunction,  java.io.Serializable{
    private IRegressor _regressor;
    private IRegressor _residuals;
    private boolean _isTrained = false;


    public AbsoluteErrorNonconformityFunction(IRegressor regressor) {
        _regressor = regressor;
    }

    @Override
    public void fit(DoubleMatrix2D x, double[] y) {

        if (_regressor instanceof GaussianProcessRegressor) {
            int folds = 3;
            double[] sigmaOptions = {0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0};
            double[] lambdaOptions = {0.0001, 0.001, 0.01, 0.1};
            ((GaussianProcessRegressor)_regressor).gridSearch(x, y, folds, sigmaOptions, lambdaOptions);
            double[] newY = computeResiduals(x, y);
            System.out.println("RESIDUALS :");
            for (int i = 0; i < newY.length; i++) {
                System.out.println(newY[i]);
            }
            _residuals = _regressor.fitNew(x, newY);
            /*_residuals = new GaussianProcessRegressor();
            ((GaussianProcessRegressor)_residuals).gridSearch(x, newY, folds, sigmaOptions, lambdaOptions);*/
        }
        else if (_regressor instanceof NN4jRegressor) {
            
            double[] learningRates = {0.001, 0.01, 0.1};
            int[] hiddenLayerSizes = {50, 100, 150};
            int[] numEpochs = {50, 100, 150};
            ((NN4jRegressor)_regressor).gridSearch(x, y, 3, learningRates, hiddenLayerSizes, numEpochs);
            double[] newY = computeResiduals(x, y);
            System.out.println("RESIDUALS :");
            for (int i = 0; i < newY.length; i++) {
                System.out.println(newY[i]);
            }
            _residuals = new NN4jRegressor();
            ((NN4jRegressor)_residuals).gridSearch(x, newY, 3, learningRates, hiddenLayerSizes, numEpochs);

        }
        else {
            _regressor.fit(x, y);
            double[] newY = computeResiduals(x, y);
            System.out.println("RESIDUALS :");
            for (int i = 0; i < newY.length; i++) {
                System.out.println(newY[i]);
            }
            _residuals = _regressor.fitNew(x, newY);
        }
        

        /*int numIterations = 50;
        double sigmaMin = 0.1;
        double sigmaMax = 10000;
        double lambdaMin = 0.0001;
        double lambdaMax = 0.1; 

        ((GaussianProcessRegressor)_regressor).randomSearch(x, y, folds, numIterations, sigmaMin, sigmaMax, lambdaMin, lambdaMax);*/

        
        // x + y true + y predicted => error
        // create new model for residuals with gridsearch 
        // residual  
        //_residuals = _regressor.fitNew(x, newY);
        /*_residuals = new GaussianProcessRegressor();
        ((GaussianProcessRegressor)_residuals).gridSearch(x, newY, folds, sigmaOptions, lambdaOptions);*/
 
        _isTrained = _residuals != null;

        
    }

    @Override
    public boolean isTrained() {
        return _isTrained;
    }

    @Override
    public double calculateNonConformityScore(DoubleMatrix1D instance, double label) {
        double[] prediction = predictWithUncertainty(instance);
        double predictedValue = prediction[0];
        double uncertainty = prediction[1]; 
        return Math.abs(label - predictedValue) / uncertainty;
    }

    @Override
    public double predict(DoubleMatrix1D instance) {
        return _regressor.predict(instance);
    }

    public double[] predictWithUncertainty(DoubleMatrix1D instance){
        double prediction = _regressor.predict(instance);
        double uncertainty = _residuals.predict(instance);
        if(uncertainty == 0.0) {
            uncertainty = 0.01;
        }
        return new double[]{prediction, uncertainty};
    }

    public double[] computeResiduals(DoubleMatrix2D x, double[] y) {
        double[] newY = new double[y.length];
        for (int i = 0; i < y.length; i++) {
            newY[i] = Math.abs(y[i] - predict(x.viewRow(i)));
        }
        return newY;
    }

    @Override
    public int getAttributeCount() {
        return _regressor.getAttributeCount();
    }

    @Override
    public DoubleMatrix1D nativeStorageTemplate() {
        return _regressor.nativeStorageTemplate();
    }

    @Override
    public DoubleMatrix2D predict(AbstractMatrix2D x, double significance) {
        System.out.println("To implement");
        return null; 
    }
}

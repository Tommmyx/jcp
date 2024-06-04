package se.hb.jcp.nc;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.AbstractMatrix2D;
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
        _regressor.fit(x, y);
        _isTrained = true;
        double[] newY = computeResiduals(x, y);
        for(int i = 0; i < newY.length; i++) {
            System.out.println(newY[i]);
        }
        _residuals = _regressor.fitNew(x, newY);
        _isTrained = _residuals != null;
    }

    @Override
    public boolean isTrained() {
        return _isTrained;
    }

    @Override
    public double calculateNonConformityScore(DoubleMatrix1D instance, double label) {
        double prediction = _regressor.predict(instance);
        return Math.abs(label - prediction);
    }

    @Override
    public double predict(DoubleMatrix1D instance) {
        return _regressor.predict(instance);
    }

    public double[] predictWithUncertainty(DoubleMatrix1D instance){
        double prediction = predict(instance);
        double uncertainty = _residuals.predict(instance);
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

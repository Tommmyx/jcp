package se.hb.jcp.cp;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.SparseDoubleMatrix1D;
import se.hb.jcp.nc.IRegressionNonconformityFunction;
import se.hb.jcp.util.ParallelizedAction;

import java.io.Serializable;
import java.util.Arrays;
import java.util.stream.IntStream;

public class GaussianProcessConformalRegressor implements IConformalRegressor, Serializable {

    private static final boolean PARALLEL = false;

    private IRegressionNonconformityFunction _nc;
    private double[] _calibrationScores;

    public GaussianProcessConformalRegressor(IRegressionNonconformityFunction nc) {
        _nc = nc;
    }

    public void fit(DoubleMatrix2D xtr, double[] ytr, DoubleMatrix2D xcal, double[] ycal) {
        _nc.fit(xtr, ytr);    
        _calibrationScores = IntStream.range(0, xcal.rows())
                                  .parallel()
                                  .mapToDouble(i -> {
                                      DoubleMatrix1D instance = xcal.viewRow(i);
                                      double[] prediction = _nc.predictWithUncertainty(instance);
                                      double predictedValue = prediction[0];
                                      double uncertainty = prediction[1];
                                      return Math.abs(ycal[i] - predictedValue) / uncertainty;
                                  })
                                  .sorted()
                                  .toArray();
    }

    /*public void calibrate(DoubleMatrix2D xcal, double[] ycal) {
        int n = xcal.rows();
        _calibrationScores = new double[n];

        for (int i = 0; i < n; i++) {
            DoubleMatrix1D instance = xcal.viewRow(i);
            double[] prediction = _nc.predictWithUncertainty(instance);
            double predictedValue = prediction[0];
            double uncertainty = prediction[1];
            _calibrationScores[i] = Math.abs(ycal[i] - predictedValue) / uncertainty;
        }

        Arrays.sort(_calibrationScores);
    }*/

    public double calculateQuantile(double alpha) {
        int n = _calibrationScores.length;
        int index = (int) Math.ceil((1 - alpha) * (n + 1)) - 1;
        System.out.println("CALIB " + _calibrationScores[Math.max(0, Math.min(index, n - 1))]);
        return _calibrationScores[Math.max(0, Math.min(index, n - 1))];
    }

    public double[] predictIntervals(DoubleMatrix1D x, double alpha) {
        double[] prediction = _nc.predictWithUncertainty(x);
        double predictedValue = prediction[0];
        double uncertainty = prediction[1];
        double qhat = calculateQuantile(alpha);
        return new double[]{predictedValue - qhat * uncertainty, predictedValue + qhat * uncertainty};
    }
    
    public double[][] predictIntervals(DoubleMatrix2D x, double confidence) {
        int n = x.rows();
        double[][] intervals = new double[n][2];
        if (!PARALLEL) {
            for (int i = 0; i < n; i++) {
                intervals[i] = predictIntervals(x.viewRow(i), confidence);
            }
        } else {
            PredictIntervalsAction all = new PredictIntervalsAction(x, intervals, confidence, 0, n);
            all.start();
        }
        return intervals;
    }

    @Override
    public boolean isTrained() {
        return _calibrationScores != null;
    }

    @Override
    public int getAttributeCount() {
        if (getNonconformityFunction() != null) {
            return getNonconformityFunction().getAttributeCount();
        } else {
            return -1;
        }
    }

    @Override
    public DoubleMatrix1D nativeStorageTemplate() {
        if (getNonconformityFunction() != null) {
            return getNonconformityFunction().nativeStorageTemplate();
        } else {
            return new SparseDoubleMatrix1D(0);
        }
    }

    public IRegressionNonconformityFunction getNonconformityFunction() {
        return _nc;
    }

    class PredictIntervalsAction extends ParallelizedAction {
        DoubleMatrix2D _x;
        double[][] _intervals;
        double _confidence;

        public PredictIntervalsAction(DoubleMatrix2D x, double[][] intervals, double confidence, int first, int last) {
            super(first, last);
            _x = x;
            _intervals = intervals;
            _confidence = confidence;
        }

        @Override
        protected void compute(int i) {
            _intervals[i] = predictIntervals(_x.viewRow(i), _confidence);
        }

        @Override
        protected ParallelizedAction createSubtask(int first, int last) {
            return new PredictIntervalsAction(_x, _intervals, _confidence, first, last);
        }
    }

    class CalculateNCScoresAction extends ParallelizedAction {
        DoubleMatrix2D _x;
        double[] _y;
        double[] _nonConformityScores;

        public CalculateNCScoresAction(DoubleMatrix2D x, double[] y, double[] nonConformityScores, int first, int last) {
            super(first, last);
            _x = x;
            _y = y;
            _nonConformityScores = nonConformityScores;
        }

        @Override
        protected void compute(int i) {
            DoubleMatrix1D instance = _x.viewRow(i);
            _nonConformityScores[i] = _nc.calculateNonConformityScore(instance, _y[i]);
        }

        @Override
        protected ParallelizedAction createSubtask(int first, int last) {
            return new CalculateNCScoresAction(_x, _y, _nonConformityScores, first, last);
        }
    }

}


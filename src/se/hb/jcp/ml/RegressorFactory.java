// JCP - Java Conformal Prediction framework
// Copyright (C) 2014 - 2015  Anders Gidenstam
//
// This library is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published
// by the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
package se.hb.jcp.ml;

import org.json.JSONObject;

public class RegressorFactory {
    private static final RegressorFactory _theInstance =
        new RegressorFactory();
    //add regressor
    private static final String[] _regressorNames =
    {
        "se.hb.jcp.bindings.libsvm.SVMClassifier",
    };  
    private RegressorFactory()
    {
    }

    public String[] getRegressorTypes()
    {
        return _regressorNames;
    }

    public IClassifier createRegressor(int type)
    {
        JSONObject config = new JSONObject();
        return createRegressor(type, config);
    }

    public IClassifier createRegressor(int type, JSONObject config)
    {
        switch (type) {
        case 0:
            return new se.hb.jcp.bindings.libsvm.SVMClassifier(config);
        default:
            throw new UnsupportedOperationException("Unknown classifier type.");
        }
    }

    public static RegressorFactory getInstance()
    {
        return _theInstance;
    }


}
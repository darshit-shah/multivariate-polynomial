(function (exports) {
    // JScript source code
    "use strict";
    var numeric = require('numericjs');

    exports.MPoly = {
        Inverse: function (x) {
            var svd = numeric.svd(x);
            var lastNonZero = svd.S.length;
            while (svd.S[lastNonZero - 1] == 0) {
                lastNonZero--;
            }
            for (var i = 0; i < svd.S.length; i++) {
                var v = svd.S[i];
                svd.S[i] = Array.apply(null, new Array(svd.S.length)).map(Number.prototype.valueOf, 0);
                svd.S[i][i] = v;
            }
            var newS = new Array(lastNonZero);
            for (var i = 0; i < newS.length; i++) {
                var v = svd.S[i][i];
                newS[i] = Array.apply(null, new Array(newS.length)).map(Number.prototype.valueOf, 0);
                newS[i][i] = v;
            }
            newS = numeric.inv(newS);
            for (var i = 0; i < newS.length; i++) {
                for (var j = 0; j < newS[i].length; j++) {
                    svd.S[i][j] = newS[i][j];
                }
            }
            return numeric.dot((numeric.dot(svd.V, svd.S)), this.Transpose(svd.U));
        },

        Regression: function (predict, otherData, degree) {
            var multiVariateSize = otherData.length;

            var otherDataPower = new Array();
            for (var j = 0; j <= degree; j++) {
                var currDegree = j;
                for (var j1 = 0; j1 <= currDegree; j1++) {
                    otherDataPower.push([(currDegree - j1), j1]);
                }
            }

            var X = new Array(predict.length);
            var Y = new Array(predict.length);
            for (var i = 0; i < X.length; i++) {
                Y[i] = [predict[i]];
                X[i] = new Array(otherDataPower.length);
                for (var j = 0; j < otherDataPower.length; j++) {
                    X[i][j] = Math.pow(otherData[0][i], otherDataPower[j][0]) * Math.pow(otherData[1][i], otherDataPower[j][1]);
                }
            }


            //console.log(otherDataPower, otherDataPower.length, X);

            //        var arrCol = [];
            //        for (var i = 0; i < X[0].length; i++) {
            //            var arrRow = [];
            //            for (var j = 0; j < X.length; j++) {
            //                arrRow.push(X[j][i]);
            //            }
            //            arrCol.push(arrRow.join());
            //        }
            //        console.log(arrCol.join("|"));

            var storage = {};
            storage.X = X;
            //console.log(storage.X);
            storage.Y = Y;
            //console.log(storage.Y);
            storage.Xt = this.Transpose(storage.X);
            //console.log(storage.Xt);
            storage.XtX = numeric.dot(storage.Xt, storage.X);
            //console.log('XtX', storage.XtX);
            storage.XtY = numeric.dot(storage.Xt, storage.Y);
            //console.log(storage.XtY);
            storage.Yt = this.Transpose(storage.Y);
            //console.log(storage.Yt);
            storage.YtY = numeric.dot(storage.Yt, storage.Y);
            //console.log(storage.YtY);
            storage.Inv_XtX = this.Inverse(storage.XtX);
            //console.log('Inv_XtX', storage.Inv_XtX);
            storage.Beta = numeric.dot(storage.Inv_XtX, storage.XtY);
            //console.log(storage.Beta);

            storage.Yreg = numeric.dot(storage.X, storage.Beta);
            storage.profile = this.profile(storage.Y, storage.Yreg);
            //console.log(storage.rSquare, storage.Beta);
            //        if (storage.profile.rSquare < 0 && degree > 1) {
            //            //console.log([degree, storage.rSquare]);
            //            storage = {};
            //            return this.MultivariateRegression(predict, otherData, degree - 1);
            //        }
            //        else {
            //        console.log(degree, storage.profile.rSquare);
            //        for (var j = 0; j < otherDataPower.length; j++) {
            //            console.log(otherDataPower[j], storage.Beta[j]);
            //        }

            return { degree: degree, power: otherDataPower, beta: storage.Beta, profile: storage.profile, data: storage };
            //        }
        },
        AnomalyDetection: function(predict, otherData, degree, useFirstNPoints){
        	var AnomalyData=[];
        	var localPredict=[];
        	var localOtherData=new Array(otherData.length);
        	for(var i=0;i<predict.length;i++){
        		if(i>=useFirstNPoints){
        			var orgReg = this.Regression(localPredict, localOtherData, degree);
        		}
        		localPredict.push(predict[i]);
        		for (var j = 0; j < otherData.length; j++) {
        			if(localOtherData[j]==null){
        				localOtherData[j] = [];
        			}
        			localOtherData[j].push(otherData[j][i]);
        		}
        		if(i>=useFirstNPoints){
        			//console.log(['start ', i, predictLocal, otherDataLocal, degree]);
              var newReg = this.Regression(localPredict, localOtherData, degree);
              //console.log(['completed ', i]);
              var BminB = numeric.sub(orgReg.beta, newReg.beta);
              //console.log(['BminB', BminB]);
              var BminBT = numeric.transpose(BminB);
              //console.log(['BminBT', BminBT]);
              var o = numeric.dot(BminBT, orgReg.data.XtX);
              //console.log(['o', o]);
              o = numeric.dot(o, BminB);
              var p = otherData.length;
              //console.log(['o', o[0][0], ((1 + p) * orgReg.profile.MSE)]);
              var n = localPredict.length;
              var check = o[0][0] / ((1 + p) * orgReg.profile.MSE);
              if (check > (4 / n)) {
              	AnomalyData.push(1);
              }
              else {
              	AnomalyData.push(0);
              }
        		}
        		else{
        			AnomalyData.push(-1);
        		}
        	}
        	return AnomalyData
        },
        MultivariateRegression: function (predict, otherData, degree, removeCooksDistance) {
            var orgReg = this.Regression(predict, otherData, degree);
            if (removeCooksDistance != true) {
                return orgReg;
            }
            else {
                var outliers = [];
                var D = new Array(predict.length);
                for (var i = 0; i < predict.length; i++) {
                    var predictLocal = predict.slice(0);
                    predictLocal.splice(i, 1);
                    var otherDataLocal = otherData.slice(0);
                    for (var j = 0; j < otherDataLocal.length; j++) {
                        otherDataLocal[j] = otherData[j].slice(0);
                        otherDataLocal[j].splice(i, 1);
                    }
                    //console.log(['start ', i, predictLocal, otherDataLocal, degree]);
                    var newReg = this.Regression(predictLocal, otherDataLocal, degree);
                    //console.log(['completed ', i]);
                    var BminB = numeric.sub(orgReg.beta, newReg.beta);
                    //console.log(['BminB', BminB]);
                    var BminBT = numeric.transpose(BminB);
                    //console.log(['BminBT', BminBT]);
                    var o = numeric.dot(BminBT, orgReg.data.XtX);
                    //console.log(['o', o]);
                    o = numeric.dot(o, BminB);
                    var p = otherData.length;
                    //console.log(['o', o[0][0], ((1 + p) * orgReg.profile.MSE)]);
                    D[i] = o[0][0] / ((1 + p) * orgReg.profile.MSE);
                    //console.log(D[i]);
                }
                var n = predict.length;
                //console.log(D, 4 / n);
                for (var i = 0; i < predict.length; i++) {
                    if (D[i] > (4 / n)) {
                        //console.log(['di', i, D[i], 4 / n]);
                        var lPredict = predict.splice(i, 1)
                        //console.log(['Predict', lPredict]);
                        for (var j = 0; j < otherData.length; j++) {
                            var lother = otherData[j].splice(i, 1);
                            //console.log(['otherData', j, lother]);
                        }
                        D.splice(i, 1);
                        i--;
                    }
                }
                //console.log(['Final ', predict, otherData, degree]);
                //console.log(predict.length);
                var finalReg = this.Regression(predict, otherData, degree);
                //console.log(['completed ']);
                return finalReg;
            }
        },
        profile: function (original, regression) {
            var SSReg = 0;
            var SSTot = 0;
            var SSAvg = numeric.sum(original) / original.length;
            for (var i = 0; i < original.length; i++) {
                SSReg += Math.pow(original[i] - regression[i], 2);
                SSTot += Math.pow(original[i] - SSAvg, 2);
            }
            //console.log(SSReg, SSTot, SSAvg)
            var profile = { rSquare: 1 - (SSReg / SSTot), MSE: SSReg / original.length };
            return profile;
        },
        Transpose: function (x) {
            var xt = NaN;
            if (!Array.isArray(x)) {
                xt = x;
            }
            else if (!Array.isArray(x[0])) {
                xt = new Array(x.length);
                for (var i = 0; i < x.length; i++) {
                    xt[i] = [x[i]];
                }
            }
            else if (x[0].length == 1) {
                xt = new Array(x.length);
                for (var i = 0; i < x.length; i++) {
                    xt[i] = x[i][0];
                }
            }
            else {
                var xt = new Array(x[0].length);
                for (var i = 0; i < x[0].length; i++) {
                    xt[i] = new Array(x.length);
                }
                for (var row = 0; row < x.length; row++) {
                    if (x[row].length != x[0].length) {
                        xt = NaN;
                        break;
                    }
                    for (var col = 0; col < x[row].length; col++) {
                        xt[col][row] = x[row][col];
                    }
                }
            }
            return xt;
        },
        Predict: function (Regression, otherData) {
            var degree = Regression.degree, otherDataPower = Regression.power, beta = Regression.beta;
            var values = 0;
            for (var i = 0; i < otherDataPower.length; i++) {
                values += Math.pow(otherData[0], otherDataPower[i][0]) * Math.pow(otherData[1], otherDataPower[i][1]) * beta[i];
            }
            //console.log(values);
            return values;
        }
    }
		
		/*
		Example 1
    var _degree = 2;

    var volumes = [80.4166666666666, 105.779761904761, 116.680952380952, 126.268253968253, 127.449801587301, 146.407738095238, 150.133928571428, 155.717261904761, 150.681547619047, 137.8125, 119.47619047619, 97.5178571428571, 89.1916666666666, 86.6535714285714, 69.8619047619047, 71.1714285714285, 75.43];
    var prices = [-7.15074561999995, -2.38835662428567, -0.749820143968217, 0.523431953174622, -0.0262293164682508, -1.97562155059523, -1.09659232916665, -0.629858449999991, 2.0700822547619, 1.96204390535713, 2.08745380059523, 1.96456482023808, 2.39867362619046, -0.282910316666667, -0.318768846428573, -1.61195144821429, -0.43086080059525];
    var margins = [0.666667500000016, 0.333333750000008, -0.541667083333322, -1.06944529761904, -2.04166738095237, -0.56349208333332, -0.862103035714274, 0.683035714285722, -0.634424761904767, -1.44692476190476, -2.66269857142858, -1.72619047619048, -0.441468255357146, -0.274801588690479, -1.18452380535715, -1.62400793095239, -1.81448412142858];
    var ts = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17];

		var anomalyData = exports.MPoly.AnomalyDetection(volumes, [ts, prices, margins], _degree, 10);
		console.log(anomalyData);
		*/
		
		/*
		Example 2
    var equation = exports.MPoly.MultivariateRegression(volumes, [ts, prices, margins], _degree, false);
    console.log(equation.profile)
    for (var i = 0; i < volumes.length; i++) {
        var vol = exports.MPoly.Predict(equation, [(i + 1), prices[i], margins[i]]);
        console.log(parseInt(prices[i]), parseInt(margins[i]), parseInt(volumes[i]), parseInt(vol), parseInt((volumes[i] - vol) * 100 / volumes[i]));
    }

    equation = exports.MPoly.MultivariateRegression(volumes, [prices, margins, ts], _degree, true);
    console.log(equation.profile)
    for (var i = 0; i < volumes.length; i++) {
        var vol = exports.MPoly.Predict(equation, [prices[i], margins[i], (i + 1)]);
        console.log(parseInt(prices[i]), parseInt(margins[i]), parseInt(volumes[i]), parseInt(vol), parseInt((volumes[i] - vol) * 100 / volumes[i]));
    }
    */

})(typeof exports === 'undefined' ? this['MPoly'] = {} : exports);
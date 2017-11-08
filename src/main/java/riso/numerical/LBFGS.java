package riso.numerical;

public class LBFGS {

	public static class ExceptionWithIflag extends Exception {
		public int iflag;

		public ExceptionWithIflag(int i, String s) {
			super(s);
			iflag = i;
		}

		public String toString() {
			return getMessage() + " (iflag == " + iflag + ")";
		}
	}

	public static double gtol = 0.9;

	public static double stpmin = 1e-20;

	public static double stpmax = 1e20;

	public static double[] solution_cache = null;
	public static double gnorm = 0, stp1 = 0, ftol = 0, stp[] = new double[1], ys = 0, yy = 0, sq = 0, yr = 0, beta = 0,
			xnorm = 0;
	public static int iter = 0, nfun = 0, point = 0, ispt = 0, iypt = 0, maxfev = 0, info[] = new int[1], bound = 0,
			npt = 0, cp = 0, i = 0, nfev[] = new int[1], inmc = 0, iycn = 0, iscn = 0;
	public static boolean finish = false;

	static double[] w = null;

	public static int nfevaluations() {
		return nfun;
	}

	public static void lbfgs(int n, int m, double[] x, double f, double[] g, boolean diagco, double[] diag,
			int[] iprint, double eps, double xtol, int[] iflag) throws ExceptionWithIflag {
		boolean execute_entire_while_loop = false;

		if (w == null || w.length != n * (2 * m + 1) + 2 * m) {
			w = new double[n * (2 * m + 1) + 2 * m];
		}

		if (iflag[0] == 0) {

			solution_cache = new double[n];
			System.arraycopy(x, 0, solution_cache, 0, n);

			iter = 0;

			if (n <= 0 || m <= 0) {
				iflag[0] = -3;
				throw new ExceptionWithIflag(iflag[0], "Improper input parameters  (n or m are not positive.)");
			}

			if (gtol <= 0.0001) {
				System.err.println("LBFGS.lbfgs: gtol is less than or equal to 0.0001. It has been reset to 0.9.");
				gtol = 0.9;
			}

			nfun = 1;
			point = 0;
			finish = false;

			if (diagco) {
				for (i = 1; i <= n; i += 1) {
					if (diag[i - 1] <= 0) {
						iflag[0] = -2;
						throw new ExceptionWithIflag(iflag[0], "The " + i
								+ "-th diagonal element of the inverse hessian approximation is not positive.");
					}
				}
			} else {
				for (i = 1; i <= n; i += 1) {
					diag[i - 1] = 1;
				}
			}
			ispt = n + 2 * m;
			iypt = ispt + n * m;

			for (i = 1; i <= n; i += 1) {
				w[ispt + i - 1] = -g[i - 1] * diag[i - 1];
			}

			gnorm = Math.sqrt(ddot(n, g, 0, 1, g, 0, 1));
			stp1 = 1 / gnorm;
			ftol = 0.0001;
			maxfev = 20;

			if (iprint[1 - 1] >= 0)
				lb1(iprint, iter, nfun, gnorm, n, m, x, f, g, stp, finish);

			execute_entire_while_loop = true;
		}

		while (true) {
			if (execute_entire_while_loop) {
				iter = iter + 1;
				info[0] = 0;
				bound = iter - 1;
				if (iter != 1) {
					if (iter > m)
						bound = m;
					ys = ddot(n, w, iypt + npt, 1, w, ispt + npt, 1);
					if (!diagco) {
						yy = ddot(n, w, iypt + npt, 1, w, iypt + npt, 1);

						for (i = 1; i <= n; i += 1) {
							diag[i - 1] = ys / yy;
						}
					} else {
						iflag[0] = 2;
						return;
					}
				}
			}

			if (execute_entire_while_loop || iflag[0] == 2) {
				if (iter != 1) {
					if (diagco) {
						for (i = 1; i <= n; i += 1) {
							if (diag[i - 1] <= 0) {
								iflag[0] = -2;
								throw new ExceptionWithIflag(iflag[0], "The " + i
										+ "-th diagonal element of the inverse hessian approximation is not positive.");
							}
						}
					}
					cp = point;
					if (point == 0)
						cp = m;
					w[n + cp - 1] = 1 / ys;

					for (i = 1; i <= n; i += 1) {
						w[i - 1] = -g[i - 1];
					}

					cp = point;

					for (i = 1; i <= bound; i += 1) {
						cp = cp - 1;
						if (cp == -1)
							cp = m - 1;
						sq = ddot(n, w, ispt + cp * n, 1, w, 0, 1);
						inmc = n + m + cp + 1;
						iycn = iypt + cp * n;
						w[inmc - 1] = w[n + cp + 1 - 1] * sq;
						daxpy(n, -w[inmc - 1], w, iycn, 1, w, 0, 1);
					}

					for (i = 1; i <= n; i += 1) {
						w[i - 1] = diag[i - 1] * w[i - 1];
					}

					for (i = 1; i <= bound; i += 1) {
						yr = ddot(n, w, iypt + cp * n, 1, w, 0, 1);
						beta = w[n + cp + 1 - 1] * yr;
						inmc = n + m + cp + 1;
						beta = w[inmc - 1] - beta;
						iscn = ispt + cp * n;
						daxpy(n, beta, w, iscn, 1, w, 0, 1);
						cp = cp + 1;
						if (cp == m)
							cp = 0;
					}

					for (i = 1; i <= n; i += 1) {
						w[ispt + point * n + i - 1] = w[i - 1];
					}
				}

				nfev[0] = 0;
				stp[0] = 1;
				if (iter == 1)
					stp[0] = stp1;

				for (i = 1; i <= n; i += 1) {
					w[i - 1] = g[i - 1];
				}
			}

			Mcsrch.mcsrch(n, x, f, g, w, ispt + point * n, stp, ftol, xtol, maxfev, info, nfev, diag);///

			if (info[0] == -1) {
				iflag[0] = 1;
				return;
			}

			if (info[0] != 1) {
				iflag[0] = -1;
				throw new ExceptionWithIflag(iflag[0],
						"Line search failed. See documentation of routine mcsrch. Error return of line search: info = "
								+ info[0]
								+ " Possible causes: function or gradient are incorrect, or incorrect tolerances.");
			}

			nfun = nfun + nfev[0];
			npt = point * n;

			for (i = 1; i <= n; i += 1) {
				w[ispt + npt + i - 1] = stp[0] * w[ispt + npt + i - 1];
				w[iypt + npt + i - 1] = g[i - 1] - w[i - 1];
			}

			point = point + 1;
			if (point == m)
				point = 0;

			gnorm = Math.sqrt(ddot(n, g, 0, 1, g, 0, 1));
			xnorm = Math.sqrt(ddot(n, x, 0, 1, x, 0, 1));
			xnorm = Math.max(1.0, xnorm);

			if (gnorm / xnorm <= eps)
				finish = true;

			if (iprint[1 - 1] >= 0)
				lb1(iprint, iter, nfun, gnorm, n, m, x, f, g, stp, finish);

			System.arraycopy(x, 0, solution_cache, 0, n);

			if (finish) {
				iflag[0] = 0;
				return;
			}

			execute_entire_while_loop = true;
		}
	}

	public static void lb1(int[] iprint, int iter, int nfun, double gnorm, int n, int m, double[] x, double f,
			double[] g, double[] stp, boolean finish) {
		int i;

		if (iter == 0) {
			System.out.println("*************************************************");
			System.out.println("  n = " + n + "   number of corrections = " + m + "\n       initial values");
			System.out.println(" f =  " + f + "   gnorm =  " + gnorm);
			if (iprint[2 - 1] >= 1) {
				System.out.print(" vector x =");
				for (i = 1; i <= n; i++)
					System.out.print("  " + x[i - 1]);
				System.out.println("");

				System.out.print(" gradient vector g =");
				for (i = 1; i <= n; i++)
					System.out.print("  " + g[i - 1]);
				System.out.println("");
			}
			System.out.println("*************************************************");
			System.out.println("\ti\tnfn\tfunc\tgnorm\tsteplength");
		} else {
			if ((iprint[1 - 1] == 0) && (iter != 1 && !finish))
				return;
			if (iprint[1 - 1] != 0) {
				if ((iter - 1) % iprint[1 - 1] == 0 || finish) {
					if (iprint[2 - 1] > 1 && iter > 1)
						System.out.println("\ti\tnfn\tfunc\tgnorm\tsteplength");
					System.out.println("\t" + iter + "\t" + nfun + "\t" + f + "\t" + gnorm + "\t" + stp[0]);
				} else {
					return;
				}
			} else {
				if (iprint[2 - 1] > 1 && finish)
					System.out.println("\ti\tnfn\tfunc\tgnorm\tsteplength");
				System.out.println("\t" + iter + "\t" + nfun + "\t" + f + "\t" + gnorm + "\t" + stp[0]);
			}
			if (iprint[2 - 1] == 2 || iprint[2 - 1] == 3) {
				if (finish) {
					System.out.print(" final point x =");
				} else {
					System.out.print(" vector x =  ");
				}
				for (i = 1; i <= n; i++)
					System.out.print("  " + x[i - 1]);
				System.out.println("");
				if (iprint[2 - 1] == 3) {
					System.out.print(" gradient vector g =");
					for (i = 1; i <= n; i++)
						System.out.print("  " + g[i - 1]);
					System.out.println("");
				}
			}
			if (finish)
				System.out.println(" The minimization terminated without detecting errors. iflag = 0");
		}
		return;
	}

	public static void daxpy(int n, double da, double[] dx, int ix0, int incx, double[] dy, int iy0, int incy) {
		int i, ix, iy, m, mp1;

		if (n <= 0)
			return;

		if (da == 0)
			return;

		if (!(incx == 1 && incy == 1)) {
			ix = 1;
			iy = 1;

			if (incx < 0)
				ix = (-n + 1) * incx + 1;
			if (incy < 0)
				iy = (-n + 1) * incy + 1;

			for (i = 1; i <= n; i += 1) {
				dy[iy0 + iy - 1] = dy[iy0 + iy - 1] + da * dx[ix0 + ix - 1];
				ix = ix + incx;
				iy = iy + incy;
			}

			return;
		}

		m = n % 4;
		if (m != 0) {
			for (i = 1; i <= m; i += 1) {
				dy[iy0 + i - 1] = dy[iy0 + i - 1] + da * dx[ix0 + i - 1];
			}

			if (n < 4)
				return;
		}

		mp1 = m + 1;
		for (i = mp1; i <= n; i += 4) {
			dy[iy0 + i - 1] = dy[iy0 + i - 1] + da * dx[ix0 + i - 1];
			dy[iy0 + i + 1 - 1] = dy[iy0 + i + 1 - 1] + da * dx[ix0 + i + 1 - 1];
			dy[iy0 + i + 2 - 1] = dy[iy0 + i + 2 - 1] + da * dx[ix0 + i + 2 - 1];
			dy[iy0 + i + 3 - 1] = dy[iy0 + i + 3 - 1] + da * dx[ix0 + i + 3 - 1];
		}
		return;
	}

	public static double ddot(int n, double[] dx, int ix0, int incx, double[] dy, int iy0, int incy) {
		double dtemp;
		int i, ix, iy, m, mp1;

		dtemp = 0;

		if (n <= 0)
			return 0;

		if (!(incx == 1 && incy == 1)) {
			ix = 1;
			iy = 1;
			if (incx < 0)
				ix = (-n + 1) * incx + 1;
			if (incy < 0)
				iy = (-n + 1) * incy + 1;
			for (i = 1; i <= n; i += 1) {
				dtemp = dtemp + dx[ix0 + ix - 1] * dy[iy0 + iy - 1];
				ix = ix + incx;
				iy = iy + incy;
			}
			return dtemp;
		}

		m = n % 5;
		if (m != 0) {
			for (i = 1; i <= m; i += 1) {
				dtemp = dtemp + dx[ix0 + i - 1] * dy[iy0 + i - 1];
			}
			if (n < 5)
				return dtemp;
		}

		mp1 = m + 1;
		for (i = mp1; i <= n; i += 5) {
			dtemp = dtemp + dx[ix0 + i - 1] * dy[iy0 + i - 1] + dx[ix0 + i + 1 - 1] * dy[iy0 + i + 1 - 1]
					+ dx[ix0 + i + 2 - 1] * dy[iy0 + i + 2 - 1] + dx[ix0 + i + 3 - 1] * dy[iy0 + i + 3 - 1]
					+ dx[ix0 + i + 4 - 1] * dy[iy0 + i + 4 - 1];
		}

		return dtemp;
	}
}

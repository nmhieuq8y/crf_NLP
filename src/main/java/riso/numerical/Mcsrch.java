package riso.numerical;

public class Mcsrch {
	public static int infoc[] = new int[1], j = 0;
	public static double dg = 0, dgm = 0, dginit = 0, dgtest = 0, dgx[] = new double[1], dgxm[] = new double[1],
			dgy[] = new double[1], dgym[] = new double[1], finit = 0, ftest1 = 0, fm = 0, fx[] = new double[1],
			fxm[] = new double[1], fy[] = new double[1], fym[] = new double[1], p5 = 0, p66 = 0, stx[] = new double[1],
			sty[] = new double[1], stmin = 0, stmax = 0, width = 0, width1 = 0, xtrapf = 0;
	public static boolean brackt[] = new boolean[1], stage1 = false;

	static double sqr(double x) {
		return x * x;
	}

	static double max3(double x, double y, double z) {
		return x < y ? (y < z ? z : y) : (x < z ? z : x);
	}

	public static void mcsrch(int n, double[] x, double f, double[] g, double[] s, int is0, double[] stp, double ftol,
			double xtol, int maxfev, int[] info, int[] nfev, double[] wa) {
		p5 = 0.5;
		p66 = 0.66;
		xtrapf = 4;

		if (info[0] != -1) {
			infoc[0] = 1;
			if (n <= 0 || stp[0] <= 0 || ftol < 0 || LBFGS.gtol < 0 || xtol < 0 || LBFGS.stpmin < 0
					|| LBFGS.stpmax < LBFGS.stpmin || maxfev <= 0)
				return;

			dginit = 0;

			for (j = 1; j <= n; j += 1) {
				dginit = dginit + g[j - 1] * s[is0 + j - 1];
			}

			if (dginit >= 0) {
				System.out.println("The search direction is not a descent direction.");
				return;
			}

			brackt[0] = false;
			stage1 = true;
			nfev[0] = 0;
			finit = f;
			dgtest = ftol * dginit;
			width = LBFGS.stpmax - LBFGS.stpmin;
			width1 = width / p5;

			for (j = 1; j <= n; j += 1) {
				wa[j - 1] = x[j - 1];
			}

			stx[0] = 0;
			fx[0] = finit;
			dgx[0] = dginit;
			sty[0] = 0;
			fy[0] = finit;
			dgy[0] = dginit;
		}

		while (true) {
			if (info[0] != -1) {
				if (brackt[0]) {
					stmin = Math.min(stx[0], sty[0]);
					stmax = Math.max(stx[0], sty[0]);
				} else {
					stmin = stx[0];
					stmax = stp[0] + xtrapf * (stp[0] - stx[0]);
				}

				stp[0] = Math.max(stp[0], LBFGS.stpmin);
				stp[0] = Math.min(stp[0], LBFGS.stpmax);

				if ((brackt[0] && (stp[0] <= stmin || stp[0] >= stmax)) || nfev[0] >= maxfev - 1 || infoc[0] == 0
						|| (brackt[0] && stmax - stmin <= xtol * stmax))
					stp[0] = stx[0];

				for (j = 1; j <= n; j += 1) {
					x[j - 1] = wa[j - 1] + stp[0] * s[is0 + j - 1];
				}

				info[0] = -1;
				return;
			}

			info[0] = 0;
			nfev[0] = nfev[0] + 1;
			dg = 0;

			for (j = 1; j <= n; j += 1) {
				dg = dg + g[j - 1] * s[is0 + j - 1];
			}

			ftest1 = finit + stp[0] * dgtest;

			if ((brackt[0] && (stp[0] <= stmin || stp[0] >= stmax)) || infoc[0] == 0)
				info[0] = 6;

			if (stp[0] == LBFGS.stpmax && f <= ftest1 && dg <= dgtest)
				info[0] = 5;

			if (stp[0] == LBFGS.stpmin && (f > ftest1 || dg >= dgtest))
				info[0] = 4;

			if (nfev[0] >= maxfev)
				info[0] = 3;

			if (brackt[0] && stmax - stmin <= xtol * stmax)
				info[0] = 2;

			if (f <= ftest1 && Math.abs(dg) <= LBFGS.gtol * (-dginit))
				info[0] = 1;

			if (info[0] != 0)
				return;

			if (stage1 && f <= ftest1 && dg >= Math.min(ftol, LBFGS.gtol) * dginit)
				stage1 = false;

			if (stage1 && f <= fx[0] && f > ftest1) {
				fm = f - stp[0] * dgtest;
				fxm[0] = fx[0] - stx[0] * dgtest;
				fym[0] = fy[0] - sty[0] * dgtest;
				dgm = dg - dgtest;
				dgxm[0] = dgx[0] - dgtest;
				dgym[0] = dgy[0] - dgtest;

				mcstep(stx, fxm, dgxm, sty, fym, dgym, stp, fm, dgm, brackt, stmin, stmax, infoc);

				fx[0] = fxm[0] + stx[0] * dgtest;
				fy[0] = fym[0] + sty[0] * dgtest;
				dgx[0] = dgxm[0] + dgtest;
				dgy[0] = dgym[0] + dgtest;
			} else {
				mcstep(stx, fx, dgx, sty, fy, dgy, stp, f, dg, brackt, stmin, stmax, infoc);
			}

			if (brackt[0]) {
				if (Math.abs(sty[0] - stx[0]) >= p66 * width1)
					stp[0] = stx[0] + p5 * (sty[0] - stx[0]);
				width1 = width;
				width = Math.abs(sty[0] - stx[0]);
			}
		}
	}

	public static void mcstep(double[] stx, double[] fx, double[] dx, double[] sty, double[] fy, double[] dy,
			double[] stp, double fp, double dp, boolean[] brackt, double stpmin, double stpmax, int[] info) {
		boolean bound;
		double gamma, p, q, r, s, sgnd, stpc, stpf, stpq, theta;

		info[0] = 0;

		if ((brackt[0] && (stp[0] <= Math.min(stx[0], sty[0]) || stp[0] >= Math.max(stx[0], sty[0])))
				|| dx[0] * (stp[0] - stx[0]) >= 0.0 || stpmax < stpmin)
			return;

		sgnd = dp * (dx[0] / Math.abs(dx[0]));

		if (fp > fx[0]) {

			info[0] = 1;
			bound = true;
			theta = 3 * (fx[0] - fp) / (stp[0] - stx[0]) + dx[0] + dp;
			s = max3(Math.abs(theta), Math.abs(dx[0]), Math.abs(dp));
			gamma = s * Math.sqrt(sqr(theta / s) - (dx[0] / s) * (dp / s));
			if (stp[0] < stx[0])
				gamma = -gamma;
			p = (gamma - dx[0]) + theta;
			q = ((gamma - dx[0]) + gamma) + dp;
			r = p / q;
			stpc = stx[0] + r * (stp[0] - stx[0]);
			stpq = stx[0] + ((dx[0] / ((fx[0] - fp) / (stp[0] - stx[0]) + dx[0])) / 2) * (stp[0] - stx[0]);
			if (Math.abs(stpc - stx[0]) < Math.abs(stpq - stx[0])) {
				stpf = stpc;
			} else {
				stpf = stpc + (stpq - stpc) / 2;
			}
			brackt[0] = true;
		} else if (sgnd < 0.0) {

			info[0] = 2;
			bound = false;
			theta = 3 * (fx[0] - fp) / (stp[0] - stx[0]) + dx[0] + dp;
			s = max3(Math.abs(theta), Math.abs(dx[0]), Math.abs(dp));
			gamma = s * Math.sqrt(sqr(theta / s) - (dx[0] / s) * (dp / s));
			if (stp[0] > stx[0])
				gamma = -gamma;
			p = (gamma - dp) + theta;
			q = ((gamma - dp) + gamma) + dx[0];
			r = p / q;
			stpc = stp[0] + r * (stx[0] - stp[0]);
			stpq = stp[0] + (dp / (dp - dx[0])) * (stx[0] - stp[0]);
			if (Math.abs(stpc - stp[0]) > Math.abs(stpq - stp[0])) {
				stpf = stpc;
			} else {
				stpf = stpq;
			}
			brackt[0] = true;
		} else if (Math.abs(dp) < Math.abs(dx[0])) {

			info[0] = 3;
			bound = true;
			theta = 3 * (fx[0] - fp) / (stp[0] - stx[0]) + dx[0] + dp;
			s = max3(Math.abs(theta), Math.abs(dx[0]), Math.abs(dp));
			gamma = s * Math.sqrt(Math.max(0, sqr(theta / s) - (dx[0] / s) * (dp / s)));
			if (stp[0] > stx[0])
				gamma = -gamma;
			p = (gamma - dp) + theta;
			q = (gamma + (dx[0] - dp)) + gamma;
			r = p / q;
			if (r < 0.0 && gamma != 0.0) {
				stpc = stp[0] + r * (stx[0] - stp[0]);
			} else if (stp[0] > stx[0]) {
				stpc = stpmax;
			} else {
				stpc = stpmin;
			}
			stpq = stp[0] + (dp / (dp - dx[0])) * (stx[0] - stp[0]);
			if (brackt[0]) {
				if (Math.abs(stp[0] - stpc) < Math.abs(stp[0] - stpq)) {
					stpf = stpc;
				} else {
					stpf = stpq;
				}
			} else {
				if (Math.abs(stp[0] - stpc) > Math.abs(stp[0] - stpq)) {
					stpf = stpc;
				} else {
					stpf = stpq;
				}
			}
		} else {
			info[0] = 4;
			bound = false;
			if (brackt[0]) {
				theta = 3 * (fp - fy[0]) / (sty[0] - stp[0]) + dy[0] + dp;
				s = max3(Math.abs(theta), Math.abs(dy[0]), Math.abs(dp));
				gamma = s * Math.sqrt(sqr(theta / s) - (dy[0] / s) * (dp / s));
				if (stp[0] > sty[0])
					gamma = -gamma;
				p = (gamma - dp) + theta;
				q = ((gamma - dp) + gamma) + dy[0];
				r = p / q;
				stpc = stp[0] + r * (sty[0] - stp[0]);
				stpf = stpc;
			} else if (stp[0] > stx[0]) {
				stpf = stpmax;
			} else {
				stpf = stpmin;
			}
		}

		if (fp > fx[0]) {
			sty[0] = stp[0];
			fy[0] = fp;
			dy[0] = dp;
		} else {
			if (sgnd < 0.0) {
				sty[0] = stx[0];
				fy[0] = fx[0];
				dy[0] = dx[0];
			}
			stx[0] = stp[0];
			fx[0] = fp;
			dx[0] = dp;
		}

		stpf = Math.min(stpmax, stpf);
		stpf = Math.max(stpmin, stpf);
		stp[0] = stpf;

		if (brackt[0] && bound) {
			if (sty[0] > stx[0]) {
				stp[0] = Math.min(stx[0] + 0.66 * (sty[0] - stx[0]), stp[0]);
			} else {
				stp[0] = Math.max(stx[0] + 0.66 * (sty[0] - stx[0]), stp[0]);
			}
		}

		return;
	}
}

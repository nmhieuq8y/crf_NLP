package org.apache.nlp.sequencelearning.crf;

import java.util.ArrayList;

import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

public class TaggerImpl {
	double cost_factor = 1.0;

	final double LOG2 = 0.69314718055;
	final double MINUS_LOG_EPSILON = 50;

	double logsumexp(double x, double y, boolean flg) {
		if (flg)
			return y;
		double vmin = Math.min(x, y);
		double vmax = Math.max(x, y);
		if (vmax > vmin + MINUS_LOG_EPSILON) {
			return vmax;
		} else {
			return vmax + Math.log(Math.exp(vmin - vmax) + 1.0);
		}
	}

	ArrayList<ArrayList<String>> xStr = new ArrayList<ArrayList<String>>();
	ArrayList<String> answerStr = new ArrayList<String>();
	ArrayList<ArrayList<Integer>> x = new ArrayList<ArrayList<Integer>>();
	ArrayList<Integer> answer = new ArrayList<Integer>();
	int xsize = 0;
	int ysize = 0;

	ArrayList<ArrayList<Double>> penalty = new ArrayList<ArrayList<Double>>();
	ArrayList<Integer> result = new ArrayList<Integer>();
	int nbest;
	double cost;
	double Z;

	ArrayList<Node> nodeList;
	ArrayList<CPath> pathList;

	Vector alpha = new DenseVector();
	Vector expected = new DenseVector();

	public TaggerImpl(ArrayList<ArrayList<String>> xStr, ArrayList<String> answerStr, ArrayList<ArrayList<Integer>> x,
			ArrayList<Integer> answer, int xsize, int ysize) {
		this.xStr = xStr;
		this.answerStr = answerStr;
		this.x = x;
		this.answer = answer;
		this.xsize = xsize;
		this.ysize = ysize;
	}

	public TaggerImpl(Vector alpha) {
		this.alpha = alpha;
	}

	public TaggerImpl() {
	}

	public void buildLattice() {
		LatticAllocate();

		if (x.isEmpty()) {
			return;
		}
		int fid = 0;
		for (int cur = 0; cur < xsize; cur++) {
			ArrayList<Integer> fvector = x.get(fid++);
			for (int j = 0; j < ysize; j++) {
				Lattic(cur, j).set(cur, j, fvector);
			}
		}
		for (int cur = 1; cur < xsize; cur++) {
			ArrayList<Integer> fvector = x.get(fid++);
			for (int j = 0; j < ysize; j++) {
				for (int k = 0; k < ysize; k++) {
					Lattic(cur, j, k).add(Lattic(cur - 1, j), Lattic(cur, k));
					Lattic(cur, j, k).fvector = fvector;
				}
			}
		}

		for (int cur = 0; cur < xsize; cur++) {
			for (int j = 0; j < ysize; j++) {
				calcCost(Lattic(cur, j));
				for (int pindex = 0; pindex < Lattic(cur, j).lpath.size(); pindex++) {
					calcCost(Lattic(cur, j).lpath.get(pindex));
				}
			}
		}

	}

	private void LatticAllocate() {
		int nodeNum = xsize * ysize;
		nodeList = new ArrayList<Node>(nodeNum);
		for (int i = 0; i < nodeNum; i++) {
			nodeList.add(new Node());
		}

		int pathNum = (xsize - 1) * ysize * ysize;
		pathList = new ArrayList<CPath>(pathNum);
		for (int i = 0; i < pathNum; i++) {
			pathList.add(new CPath());
		}

		for (int i = 0; i < xsize; i++) {
			result.add(0);
		}
	}

	private void calcCost(Node n) {
		double c = 0;
		ArrayList<Integer> fvector = n.fvector;
		for (int f : fvector) {
			c += alpha.get(f + n.y);
		}
		n.cost = cost_factor * c;
	}

	private void calcCost(CPath p) {
		double c = 0;
		ArrayList<Integer> fvector = p.fvector;
		for (int f : fvector) {
			c += alpha.get(f + p.lnode.y * ysize + p.rnode.y);
		}
		p.cost = cost_factor * c;
	}

	public void forwardbackward() {
		if (x.isEmpty()) {
			return;
		}

		for (int i = 0; i < xsize; i++) {
			for (int j = 0; j < ysize; j++) {
				Lattic(i, j).calcAlpha();
			}
		}

		for (int i = xsize - 1; i >= 0; i--) {
			for (int j = 0; j < ysize; j++) {
				Lattic(i, j).calcBeta();
			}
		}

		Z = 0.0;
		for (int j = 0; j < ysize; ++j) {
			Z = logsumexp(Z, Lattic(0, j).beta, j == 0);
		}
	}

	/**
	 * viterbiç®—æ³•
	 */
	public ArrayList<Integer> viterbi() {
		for (int i = 0; i < xsize; i++) {
			for (int j = 0; j < ysize; j++) {
				double bestc = -1e37;
				Node best = null;
				for (CPath path : Lattic(i, j).lpath) {
					double cost = path.lnode.bestCost + path.cost + Lattic(i, j).cost;
					if (cost > bestc) {
						bestc = cost;
						best = path.lnode;
					}
				}
				Lattic(i, j).prev = best;
				if (best != null) {
					Lattic(i, j).bestCost = bestc;
				} else {
					Lattic(i, j).bestCost = Lattic(i, j).cost;
				}
			}
		}

		double bestc = -1e37;
		Node best = null;
		int s = xsize - 1;
		for (int j = 0; j < ysize; j++) {
			if (bestc < Lattic(s, j).bestCost) {
				best = Lattic(s, j);
				bestc = Lattic(s, j).bestCost;
			}
		}

		result = new ArrayList<Integer>();
		for (int i = 0; i < xsize; i++) {
			result.add(0);
		}
		for (Node n = best; n != null; n = n.prev) {
			result.set(n.x, n.y);
		}
		cost = -Lattic(xsize - 1, result.get(xsize - 1)).bestCost;

		return result;
	}

	public int eval() {
		int err = 0;
		for (int i = 0; i < xsize; ++i) {
			if (answer.get(i) != result.get(i)) {
				++err;
			}
		}
		return err;
	}

	public double gradient() {
		if (x.isEmpty()) {
			return 0.0;
		}

		buildLattice();
		forwardbackward();

		double s = 0.0;
		for (int i = 0; i < xsize; i++) {
			for (int j = 0; j < ysize; j++) {
				Lattic(i, j).calcExpectation(expected, Z, ysize);
			}
		}

		for (int i = 0; i < xsize; i++) {
			Node selectedNode = Lattic(i, answer.get(i));
			ArrayList<Integer> fvector = selectedNode.fvector;

			for (int f : fvector) {
				int index = f + answer.get(i);
				expected.set(index, expected.get(index) - 1);
			}

			s += selectedNode.cost;

			ArrayList<CPath> pathAL = selectedNode.lpath;
			for (int j = 0; j < pathAL.size(); j++) {
				Node lnode = pathAL.get(j).lnode;
				Node rnode = pathAL.get(j).rnode;
				if (lnode.y == answer.get(lnode.x)) {
					ArrayList<Integer> pvector = pathAL.get(j).fvector;
					for (int f : pvector) {
						int index = f + lnode.y * ysize + rnode.y;
						expected.set(index, expected.get(index) - 1);
					}
					s += pathAL.get(j).cost;
					break;
				}
			}

		}

		viterbi();
		return Z - s;
	}

	private Node Lattic(int x, int y) {
		return nodeList.get(x * ysize + y);
	}

	private CPath Lattic(int cur, int j, int k) {
		return pathList.get((cur - 1) * ysize * ysize + j * ysize + k);
	}

	public void NodeDebug() {
		Node node;
		System.out.println("(x,y,alpha,beta,cost)");
		for (int i = 0; i < xsize; i++) {
			for (int j = 0; j < ysize; j++) {
				node = Lattic(i, j);
				System.out.println("(" + node.x + "," + node.y + "," + node.alpha + "," + node.beta + "," + node.cost
						+ "," + node.bestCost + ")");
			}
		}
	}

	public void CPathDebug() {
		System.out.println("{(lnode.x,lnode.y)->(rnode.x,rnode.y)}:cost");
		for (int cur = 0; cur < xsize; cur++) {
			for (int j = 0; j < ysize; j++) {
				ArrayList<CPath> lpathAL = Lattic(cur, j).lpath;
				for (CPath path : lpathAL) {
					System.out.println("{(" + path.lnode.x + "," + path.lnode.y + ")->(" + path.rnode.x + ","
							+ path.rnode.y + ")}:" + path.cost);
				}
			}
		}
	}

	public void ExpectationDebug() {
		System.out.println();
		System.out.println("ExpectationDebug():");
		for (int i = 0; i < expected.size(); i++) {
			System.out.println("expected[" + i + "]:" + expected.get(i));
		}
	}

}

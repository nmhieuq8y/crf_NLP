package org.apache.nlp.sequencelearning.crf;

import java.util.ArrayList;
import java.util.Set;
import java.util.TreeSet;

public class FeatureExpander {
	private static int xsize;
	private static int index;
	static String BOS[] = { "_B-1", "_B-2", "_B-3", "_B-4" };
	static String EOS[] = { "_B+1", "_B+2", "_B+3", "_B+4" };

	private Set<String> HiddenStateSet;
	private ArrayList<String> HiddenStateList;
	private FeatureTemplate featureTemplate;

	public static int getXsize() {
		return xsize;
	}

	public static void setXsize(int xsize) {
		FeatureExpander.xsize = xsize;
	}

	public FeatureTemplate getFeatureTemplate() {
		return featureTemplate;
	}

	public void setFeatureTemplate(FeatureTemplate featureTemplate) {
		this.featureTemplate = featureTemplate;
	}

	public FeatureExpander() {
		featureTemplate = new FeatureTemplate();
		HiddenStateSet = new TreeSet<String>();
		HiddenStateList = new ArrayList<String>();
	}

	public FeatureExpander(FeatureTemplate featureTemplate, int xsize) {
		HiddenStateSet = new TreeSet<String>();
		HiddenStateList = new ArrayList<String>();
		this.featureTemplate = featureTemplate;
		this.xsize = xsize;
	}

	public ArrayList<String> getHiddenStateList() {
		return HiddenStateList;
	}

	public Set<String> getHiddenStateSet() {
		return HiddenStateSet;
	}

	public void expandForConsoleTest(ArrayList<String> token_list, TaggerImpl tagger) {
		ArrayList<ArrayList<String>> tokenALAL = new ArrayList<ArrayList<String>>();
		ArrayList<String> tokenAL;
		String tempLabel1 = "P";
		String tempLabel2 = "R";
		String tempLabel3 = "V";
		String tempLabel4 = "N";
		String tempLabel5 = "CH";
		for (int i = 0; i < token_list.size(); i++) {
			tagger.answerStr.add(tempLabel1);
			tokenAL = new ArrayList<String>();
			tokenAL.add(token_list.get(i));
			if(i == 0) {
				tokenAL.add(tempLabel1);
			} else if(i == 1) {
				tokenAL.add(tempLabel2);
			} else if(i == 2) {
				tokenAL.add(tempLabel3);
			} else if(i == 3) {
				tokenAL.add(tempLabel4);
			} else if(i == 4) {
				tokenAL.add(tempLabel5);
			} else {
				tokenAL.add(tempLabel2);
			}
			tokenALAL.add(tokenAL);
		}

		for (int i = 0; i < tokenALAL.size(); i++) {
			ArrayList<String> featureAL = new ArrayList<String>();
			for (int j = 0; j < featureTemplate.unigram_templs.size(); j++) {
				StringBuffer feature = new StringBuffer();//
				if (!applyRule(feature, featureTemplate.unigram_templs.get(j), i, tokenALAL)) {
					System.out.println("unigram applyRule error");
					return;
				}
				featureAL.add(feature.toString());
			}
			tagger.xStr.add(featureAL);
		}
		for (int i = 0; i < tokenALAL.size(); i++) {
			ArrayList<String> featureAL = new ArrayList<String>();
			for (int j = 0; j < featureTemplate.bigram_templs.size(); j++) {
				StringBuffer feature = new StringBuffer();
				if (!applyRule(feature, featureTemplate.bigram_templs.get(j), i, tokenALAL)) {
					System.out.println("bigram applyRule error");
					return;
				}
				featureAL.add(feature.toString());
			}
			tagger.xStr.add(featureAL);
		}
		return;
	}

	public boolean expand(ArrayList<String> token_list, TaggerImpl tagger) {
		ArrayList<ArrayList<String>> tokenALAL = new ArrayList<ArrayList<String>>();

		int max_xsize = 0;
		int min_xsize = 999;
		String token[];
		String text;

		for (int i = 0; i < token_list.size(); i++) {
			if (token_list.get(i).equals("//CH")) {
				token = new String[] { "/", "CH" };
			} else {
				token = token_list.get(i).split("/");
			}
			if (token.length > 2) {
				tagger.answerStr.add(token[token.length - 1]);
				HiddenStateSet.add(token[token.length - 1]);
			} else if (token.length == 2) {
				tagger.answerStr.add(token[xsize - 1]);
				HiddenStateSet.add(token[xsize - 1]);
			}
			if (token.length > max_xsize) {
				max_xsize = token.length;
			}
			if (token.length < min_xsize) {
				min_xsize = token.length;
			}

			ArrayList<String> tokenAL = new ArrayList<String>();
			if (token.length > 2) {
				text = token[0];
				for (int j = 1; j < token.length - 1; j++) {
					text += "/" + token[j];
				}
				tokenAL.add(text);
				tokenAL.add(token[token.length - 1]);
			} else if (token.length == 2) {
				for (int j = 0; j < token.length; j++) {
					tokenAL.add(token[j]);
				}
			}

			tokenALAL.add(tokenAL);
		}

		for (int i = 0; i < tokenALAL.size(); i++) {
			ArrayList<String> featureAL = new ArrayList<String>();
			for (int j = 0; j < featureTemplate.unigram_templs.size(); j++) {
				StringBuffer feature = new StringBuffer();//
				if (!applyRule(feature, featureTemplate.unigram_templs.get(j), i, tokenALAL)) {
					System.out.println("unigram applyRule error");
					return false;
				}
				featureAL.add(feature.toString());
			}
			tagger.xStr.add(featureAL);
		}
		for (int i = 0; i < tokenALAL.size(); i++) {
			ArrayList<String> featureAL = new ArrayList<String>();
			for (int j = 0; j < featureTemplate.bigram_templs.size(); j++) {
				StringBuffer feature = new StringBuffer();
				if (!applyRule(feature, featureTemplate.bigram_templs.get(j), i, tokenALAL)) {
					System.out.println("bigram applyRule error");
					return false;
				}
				featureAL.add(feature.toString());
			}
			tagger.xStr.add(featureAL);
		}

		return true;
	}

	private boolean applyRule(StringBuffer feature, String tempLine, int pos,
			ArrayList<ArrayList<String>> sentenceALAL) {
		index = 0;
		for (; index < tempLine.length(); index++) {
			switch (tempLine.charAt(index)) {
			default:
				feature.append(tempLine.charAt(index));
				break;
			case '%':
				index++;
				switch (tempLine.charAt(index)) {
				case 'x':
					index++;
					String r = getIndex(tempLine, pos, sentenceALAL);// pos, sentenceALAL
					//
					if (r == null) {
						return false;
					}
					feature.append(r);
					break;
				default:
					return false;
				}
				break;

			}

		}
		return true;
	}

	private static String getIndex(String tempLine, int pos, ArrayList<ArrayList<String>> sentenceALAL) {
		if (tempLine.charAt(index) != '[') {
			return null;
		}
		index++;

		int col = 0;
		int row = 0;
		int neg = 1;
		if (tempLine.charAt(index) == '-') {
			neg = -1;
			index++;
		}

		for (; index < tempLine.length(); index++) {
			switch (tempLine.charAt(index)) {
			case '0':
			case '1':
			case '2':
			case '3':
			case '4':
			case '5':
			case '6':
			case '7':
			case '8':
			case '9':
				row = 10 * row + (tempLine.charAt(index) - '0');
				break;
			case ',':
				index++;
				return NEXT1(tempLine, pos, sentenceALAL, row, col, neg);
			default:
				return null;
			}
		}
		return null;
	}

	private static String NEXT1(String tempLine, int pos, ArrayList<ArrayList<String>> sentenceALAL, int row, int col,
			int neg) {
		for (; index < tempLine.length(); index++) {
			switch (tempLine.charAt(index)) {
			case '0':
			case '1':
			case '2':
			case '3':
			case '4':
			case '5':
			case '6':
			case '7':
			case '8':
			case '9':
				col = 10 * col + (tempLine.charAt(index) - '0');
				break;
			case ']': {
				row *= neg;
				if (row < -4 || row > 4 || col < 0 || col >= xsize) {
					return null;
				}

				int idx = pos + row;
				if (idx < 0) {
					return BOS[-idx - 1];
				}
				if (idx >= sentenceALAL.size()) {
					return EOS[idx - sentenceALAL.size()];
				}

				return sentenceALAL.get(idx).get(col);
			}

			default:
				return null;
			}
		}
		return null;
	}

	private void readLabel() {

	}
}

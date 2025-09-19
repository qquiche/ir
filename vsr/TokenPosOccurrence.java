package ir.vsr;

/**
 * Positional occurrence of a token in a specific document: count and all positions.
 * Positions are token indices after stopword removal (and stemming if enabled),
 * starting at 0 and strictly increasing.
 */
public class TokenPosOccurrence extends TokenOccurrence {

  /**
   * Sorted positions of this token in the document (in token-order units)
   */
  public int[] positions;

  public TokenPosOccurrence(DocumentReference docRef, int count, int[] positions) {
    super(docRef, count);
    this.positions = positions;
  }
}


package ir.vsr;

import java.util.*;

/**
 * Positional posting list for a token, extending basic token info with
 * positional occurrences and inherited IDF.
 */
public class TokenPosInfo extends TokenInfo {
  /**
   * Positional occurrence list (mirrors occList but with positions)
   */
  public List<TokenPosOccurrence> posOccList;

  public TokenPosInfo() {
    super();
    posOccList = new ArrayList<TokenPosOccurrence>();
  }
}



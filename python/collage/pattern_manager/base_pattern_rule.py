class BasePatternRule:
    # Specify supported operators with its constraints
    @staticmethod
    def op_rule():
        assert 0, "Need to implement"
    
    # Specify fusion rule
    @staticmethod
    def fusion_rule():
        assert 0, "Need to implement"

    # Checker method that tells if provided operators are valid
    # Uses op_rule to check single op and fuse_rule for multiple ops respectively
    def check(self, src, **kwargs):
        return self.op_rule(src) if not kwargs else self.fusion_rule(src, **kwargs)        

    # Last sanity check before pattern registration
    # By default, it always returns True
    # Override it if necessary
    @staticmethod
    def verify(pattern):
        return True
        

# Base class for pattern generator
class BasePatternGenerator:
  def __init__(self, pattern_rule):
      assert(pattern_rule is not None)
      self.pattern_rule = pattern_rule

  def generate(self, expr, **kwargs):
      assert 0, "Need to implement"
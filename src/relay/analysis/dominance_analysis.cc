#include <tvm/ir/error.h>
#include <tvm/relay/adt.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/pattern_functor.h>
#include <tvm/relay/analysis.h>

#include <stack>
#include <cassert>
#include <vector>
#include <unordered_map>
#include <unordered_set>

namespace tvm {
namespace relay {

class Graph{
 public:
  class creator;
  struct Node{
    Node(Expr _expr): expr(_expr) {}
    Expr expr;
    std::vector<Node*> parents;
  };

  std::unordered_map<Expr, Node*, ObjectPtrHash, ObjectPtrEqual> node_map;
  //std::unordered_map<const tvm::Object*, Node*> node_map;
  std::vector<Node*> post_dfs_order;
};

class Graph::creator : private ExprVisitor{
 public:
  creator(bool _is_reverse) : is_reverse(_is_reverse) {}
  Graph Prepare(const Expr& body){
    this->VisitExpr(body);
    return std::move(graph_);
  }

 private:
  bool is_reverse;
  Graph graph_;

  // update par-child relation
  void Update(Graph::Node* node, Graph::Node* parent){
    ICHECK(node && parent);

    //std::cerr << " ---- cur: " << node->expr << "\n";
    //std::cerr << "    ==> parents: " << parent->expr << "\n";
    //std::cerr << "    ==> parents: " << typeid(parent->expr).name() << "\n";

    node->parents.push_back(parent);
  }

  Graph::Node* getNode(const Expr& expr){
    if(graph_.node_map.count(expr)==0){
      graph_.node_map[expr] = new Node(expr);
    }
    ICHECK(graph_.node_map.count(expr));
    return graph_.node_map[expr];
  }

  // construct graph
  void VisitExpr_(const FunctionNode* op) final {
    Node* node = getNode(GetRef<Expr>(op));
    for (auto param : op->params) {
      Node* parent = getNode(param);
      if(is_reverse)  this->Update(parent, node);
      else            this->Update(node, parent);

    }
    ExprVisitor::VisitExpr_(op);
    graph_.post_dfs_order.push_back(node);
  }

  void VisitExpr_(const CallNode* call) final {
    Node* node = getNode(GetRef<Expr>(call));
    for (size_t i = 0; i < call->args.size(); ++i) {
      Node* parent = getNode(call->args[i]);
      if(is_reverse)  this->Update(parent, node);
      else            this->Update(node, parent);
    }
    ExprVisitor::VisitExpr_(call);
    graph_.post_dfs_order.push_back(node);
  }

  void VisitExpr_(const TupleNode* op) final {
    Node* node = getNode(GetRef<Expr>(op));
    for (const Expr& field : op->fields) {
      Node* parent = getNode(field);
      if(is_reverse)  this->Update(parent, node);
      else            this->Update(node, parent);

    }
    ExprVisitor::VisitExpr_(op);
    graph_.post_dfs_order.push_back(node);
  }

  void VisitExpr_(const TupleGetItemNode* op) final {
    auto tuple_type = op->tuple->checked_type().as<TupleTypeNode>();
    ICHECK(tuple_type);
    bool has_non_tensor = false;
    for (auto ty : tuple_type->fields) {
      if (!ty.as<TensorTypeNode>()) {
        has_non_tensor = true;
        break;
      }
    }

    Node* node = getNode(GetRef<Expr>(op));
    if (!has_non_tensor) {
      Node* parent = getNode(op->tuple);

      if(is_reverse)  this->Update(parent, node);
      else            this->Update(node, parent);

    }
    ExprVisitor::VisitExpr_(op);
    if (!has_non_tensor)  graph_.post_dfs_order.push_back(node);
  }

  void VisitExpr_(const VarNode* op) final {}
  void VisitExpr_(const ConstantNode* op) final {}
};


class DomGraph {
 public:
  class creator;
  struct Node{
    Node(Expr _expr): expr(_expr), depth(1), parent(nullptr) {}
    Expr expr;
    int depth;
    Node* parent;
  };

  std::unordered_map<Graph::Node*, Node*> node_map;
};

class DomGraph::creator {
 public:
  creator(bool _post_dom) : post_dom(_post_dom) {}
  tvm::Map<Expr, Expr> build(const Graph& graph){
    graph_ = std::move(graph);

    int sz = graph_.post_dfs_order.size();
    if(post_dom){
      // iterate in reverse post dfs order
      // Create Dom Nodes
      for(int i=sz-1;i>=0;i--){
        auto node = graph_.post_dfs_order[i];
        domgraph_.node_map[node] = new DomGraph::Node(node->expr);
      }
      // Update Dom Nodes
      for(int i=sz-1;i>=0;i--){
        update_node(graph_.post_dfs_order[i]);
      }
    }else{
      // iterate in post dfs order
      // Create Dom Nodes
      for(int i=0;i<sz;i++){
        auto node = graph_.post_dfs_order[i];
        domgraph_.node_map[node] = new DomGraph::Node(node->expr);
      }
      // Update Dom Nodes
      for(int i=0;i<sz;i++){
        update_node(graph_.post_dfs_order[i]);
      }
    }

    tvm::Map<Expr, Expr> domtree;
    for(int i=0;i<sz;i++){
      auto node = graph_.post_dfs_order[i];
      ICHECK(domgraph_.node_map.count(node));
      auto domnode = domgraph_.node_map[node]->parent;

      if(domnode)
        domtree.Set(node->expr, domnode->expr);

    }

    return std::move(domtree);
  }

 private:
  bool post_dom;
  Graph graph_;
  DomGraph domgraph_;

  DomGraph::Node* _get_lca(DomGraph::Node* lhs, DomGraph::Node* rhs){
    while(lhs != rhs){
      if(!lhs || !rhs) return nullptr;

      if(lhs->depth < rhs->depth) rhs = rhs->parent;
      else if(lhs->depth > rhs->depth) lhs = lhs->parent;
      else{
        lhs = lhs->parent;
        rhs = rhs->parent;
      }
    }
    return lhs;
  }

  DomGraph::Node* get_lca(Graph::Node* node){
    //std::cerr << "Find lca... " << node->expr << "\n";
    DomGraph::Node* dom_parent = nullptr;
    for(auto parent:node->parents){
      //std::cerr << "   - parent: " << parent->expr << "\n";
      if(parent->expr.as<ConstantNode>() || parent->expr.as<VarNode>())
        // do nothing
        continue;

      if(dom_parent)
        dom_parent = _get_lca(dom_parent, domgraph_.node_map[parent]);

      else
        dom_parent = domgraph_.node_map[parent];

      //if(dom_parent)   std::cerr << " --> " << dom_parent->expr << "\n";
      //else std::cerr << " --> None\n";
    }
    //std::cerr << "\n\n";
    return dom_parent;
  }

  void update_node(Graph::Node* node){
    assert(domgraph_.node_map.count(node));
    auto dnode = domgraph_.node_map[node];
    auto parent = get_lca(node);
    dnode->parent = parent;
    dnode->depth = (parent!=nullptr)?parent->depth+1:1;
  }

};

typedef std::unordered_set<Expr, ObjectPtrHash, ObjectPtrEqual> ExprSet;

Array<Expr> get_childs(const Expr& expr){
  Array<Expr> childs;
  if(expr.as<CallNode>()) {
    const CallNode* node = static_cast<const CallNode*>(expr.get());
    childs = node->args;
  }else if(expr.as<TupleNode>()){
    const TupleNode* node = static_cast<const TupleNode*>(expr.get());
    childs = node->fields;
  }else if(expr.as<TupleGetItemNode>()){
    const TupleGetItemNode* node = static_cast<const TupleGetItemNode*>(expr.get());
    childs.push_back(node->tuple);
  }else if(expr.as<ConstantNode>() || expr.as<VarNode>()){
    // do nothing
  }else
    ICHECK(0);
  return childs;
}

bool check(const Expr& expr, const ExprSet& matched_exprs, const ExprSet& doms_of_matched_exprs) {
  bool result = false;
  for(Expr child:get_childs(expr)){
    if(matched_exprs.count(child)){
      result = true;
      break;
    }else if(doms_of_matched_exprs.count(expr)){
      result = false;
      break;
    }else{
      result = result || check(child, matched_exprs, doms_of_matched_exprs);
    }
  }
  return result;
}


bool has_cycle(const Expr& expr, const ExprSet& matched_exprs, const ExprSet& doms_of_matched_exprs) {
  bool result = false;
  auto childs = get_childs(expr);
  if(matched_exprs.count(expr)){
    // recursively call has_cycle() for its childs
    for(Expr child:childs){
      result = result || has_cycle(child, matched_exprs, doms_of_matched_exprs);
      if(result) break;
    }
  }else{
    for(Expr child:childs){
      result = result || check(child, matched_exprs, doms_of_matched_exprs);
      if(result) break;

    }
  }
  return result;
}



TVM_REGISTER_GLOBAL("relay.analysis.dominance_analysis")
    .set_body_typed([](const Expr& expr, const bool post_dom) {
      auto g = Graph::creator(post_dom).Prepare(expr);
      return DomGraph::creator(post_dom).build(g);
    });


TVM_REGISTER_GLOBAL("relay.analysis.cycle_analysis")
    .set_body_typed([](const Expr& expr, const Array<Expr>& _matched_exprs, const Array<Expr>& _doms_of_matched_exprs) {
      ExprSet matched_exprs, doms_of_matched_exprs;
      for(auto e:_matched_exprs) matched_exprs.insert(e);
      for(auto e:_doms_of_matched_exprs) doms_of_matched_exprs.insert(e);

      return has_cycle(expr, matched_exprs, doms_of_matched_exprs);
    });




}  // namespace relay
}  // namespace tvm


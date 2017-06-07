# NNVM Overview

NNVM is a C++ library to help developers to build deep learning system.
It provides ways to construct, represent and transform computation graphs
invariant of how it is executed.

To begin with, let us start with a few stories to tell the design goals.

## Stories and Design Goals

X has built a new deep learning framework for image classification for fun,
with the modular tools like CuDNN and CUDA, it is not hard to assemble a C++ API.
However, most users like to use python/R/scala or other languages.
By registering the operators to NNVM, X can now get the graph composition
language front-end on these languages quickly without coding it up for
each type of language.

Y want to build a deep learning serving system on embedded devices.
To do that, we need to cut things off, as opposed to add new parts,
because codes such as gradient calculation multi-GPU scheduling is NOT relevant.
It is hard to build things from scratch as well, because we want to
reuse components such as memory optimization and kernel execution.
It is hard to do so in current frameworks because all these information
are tied to the operator interface. We want to be able to keep
certain part of the system we need and throw away other parts
to get the minimum system we need.

Z want to extend an existing deep learning system by adding a new feature,
say FPGA execution of some operators. To do so Z need to add a interface like ```FPGAKernel```
to the operators. E want to do another new feature that generate code for
certain subset of operations. Then interface like ```GenLLVMCode``` need to be added
to the operator. Eventually the system end up with a fat operator interface
in order to support everything (while everyone only want some part of it).

We can think more stories, as the deep learning landscape shifts to more devices
applications and scenarios. It is desirable to have different specialized
learning system to solve some problem well,

Here is a list of things we want:
- Minimum dependency
- Being able to assemble some part together while discarding some other parts
- No centralized operator interface but still allow user to provide various information about operators.

## Minimum Registration for a Symbolic Front-End
To use NNVM to build language front end, developer only need to register
minimum information about each operators.

```c++
NNVM_REGISTER_OP(add)
.describe("add two data together")
.set_num_inputs(2);

NNVM_REGISTER_OP(conv2d)
.describe("take 2d convolution of input")
.set_num_inputs(2);

NNVM_REGISTER_OP(assign)
.describe("assign second input argument to the first one")
.set_num_inputs(2);
```

Compiling the code with nnvm library. User can use the following interface
to compose the computation graph in python, like the following code.

```python
import nnvm.symbol as nn

# symbolic variable
x = nn.Variable('x')
y = nn.Variable('y')
w = nn.Variable('w')

z = nn.conv2d(nn.add(x, y), w, filter_size=(2,2), name='conv1')
```

The graph structure can be accessed in the backend. Currently python interface is supported.
But as NNVM follows the same C bridge API design as [MXNet](https://github.com/dmlc/mxnet),
which support many languages such as R/Julia/Scala/C++, more language support can be easily
moved in the future.

## Operator Attribute for More Extensions

While the minimum information provided by the operator is enough to get a front-end.
In order to do transformations and executing the graph. We need more information from each operator.
A typical difference between neural nets' computation graph and traditional LLVM IR is that
there are a lot more high level operators. We cannot fix the set of operators in the graph.

Instead developers are allowed to register attributes of operator. The attributes can include shape
inference function, whether the operator can be carried in-place etc.

This design to having an operator attribute registry is not uncommon in deep learning systems.
For example, MXNet has a ```OpProperty``` class, Tensorflow has a ```OpDef``` and Caffe2 have a ```OperatorSchema``` class.
However, the operator attribute interface listed in these frameworks only support a number of defined attributes of interest to the system.
For example, MXNet support inplace optimization decision, shape and type inference function.
If we want to extend the framework to add new type of attributes in each operator, we need to change the operator registry.
Eventually the operator interface become big and have to evolve in the centralized repo.

In NNVM, we decided to change the design and support arbitrary type of operator attributes,
without need to change the operator registry. This also echos the need of minimum interface
so that the code can be easier to share across multiple projects

User can register new attribute, such as inplace property checking function as follows.
```c++
using FInplaceOption = std::function<
  std::vector<std::pair<int, int> > (const NodeAttrs& attrs)>;

// attributes can be registered from multiple places.
NNVM_REGISTER_OP(add)
.set_num_inputs(1);

// register to tell first input can be calculate inplace with first output
NNVM_REGISTER_OP(add)
.set_attr<FInplaceOption>("FInplaceOption", [](const NodeAttrs& attrs) {
  return std::vector<std::pair<int, int> >{{0, 0}};
 });

NNVM_REGISTER_OP(exp)
.set_num_inputs(1)
.set_attr<FInplaceOption>("FInplaceOption", [](const NodeAttrs& attrs) {
  return std::vector<std::pair<int, int> >{{0, 0}};
 });
```

These attributes can be queried at arbitrary parts of the code, like the following parts.
Under the hood, each attributes are stored in a any type columnar store,
that can easily be retrieved and cast back to typed table and do quick lookups.

```c++
void MyFunction() {
  const Op* add = Op::Get("add");
  // if we need quick query, we can use static variable
  // attribute map contains attributes of all operators.
  static auto& finplace_option_map = Op::GetAttr<FInplaceOption>("FInplaceOption");

  // quick look up attribute of add, O(1) time, vector index lookup internally.
  auto add_inplace = finplace_option_tbl[add];
}
```
Besides making the code minimum, this attribute store enables decentralization of projects.
Before, all the attributes of operator have to sit on a centralized interface class.
Now, everyone can register their own attribute, take some other attribute they need from another project
without need to change the operator interface.

See [example code](../example/src/operator.cc) on how operators can be registered.

## Graph and Pass

With additional information about the operators, we can use them to get 
additional information about the graph and perform further optimizations.

Graph is the unit we manipulate in a given transformation step. 
A Graph in NNVM contains two parts:
- The computation graph structure
- A attribute map from string to any type ```map<string, shared_ptr<any> >```
The second attribute map is quite important. During the transformation process, 
we may need different kinds of information about the graph, e.g.
shapes of each tensor, types of each tensor or the storage allocation plans.

A ```Pass``` can take a graph with existing attribute information,
and transform it to the same graph with more attributes, or another graph.

We have some passes already implemented in NNVM, including symbolic differentiation,
memory planning, shape/type inference and we can support more.

## Executing the Graph

Currently, the library defines nothing on how the graph would be executed.

We have intentionally excluded the execution system from this module because we believe
that execution can be managed by another module, and there can be many ways to execute a given graph.
One can target existing runtime platforms, or even write one's own.

More importantly, the information such as memory allocation plan,
shape and type of each tensor can be used during the execution phase
to enhance performance.

We can also register more runtime related information to the operator registry,
and define a pass function to do runtime-related optimization of the graph.

## Relation to LLVM

NNVM is inspired by LLVM. However, it is at a higher level, in the sense that there are additional optimization
opportunities due to the higher-level information about operators.
On the other hand, we do believe that code generation to LLVM can be a natural extension and can benefit some of the usecases.

## Unix Philosophy in Learning Systems

There are a few existing computation graph based deep learning frameworks (e.g. Theano, Tensorflow, Caffe2, MXNet etc.).
NNVM does not intend to become another one. Instead, NNVM summarizes a module that ensures

- The graph representation is minimum, with no code dependency
- Operator attributes allow arbitrary information registeration in a unified manner
- It can target various execution layers to various front ends

We believe this is the correct approach to developing a machine learning system.

Given multiple choices in a modular system, we can pick the ones we need, and more importantly, remove the ones we do not want in our use cases.

We hope that our modularization effort will make deep learning systems research and development easy, fun and rewarding.

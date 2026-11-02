---
title: 1. Poses, Coordinate Frames, Rigid Transforms, and 3D Rotations
description: Coordinate frames and rigid transforms from first principles, with worked examples, rotation derivations, quaternion operations, and uniform sampling on spheres and SO(3).
date: 2026-09-02
lastmod: 2026-09-30
draft: false
tags:
  - machine-perception
  - geometry
  - robotics
  - notes
---

A car's LiDAR detects a pedestrian in **sensor coordinates**, but planning and navigation need **vehicle** and **world coordinates**. Rigid transforms connect these descriptions: calibration locates the sensor on the car, and localization places the car in the world. The pedestrian stays the same; only its coordinates change.

![An autonomous car using roof-mounted LiDAR to detect a pedestrian at a nighttime intersection, with coordinate frames attached to the sensor, vehicle, and world map](transforms.webp)
_The sensor, vehicle, and world each provide a reference frame for the detection._

We will build this frame-to-frame mapping, learn to invert and compose it, then explore rotation representations and random sampling. Throughout, we use **column vectors, right-handed frames, and radians unless degrees are explicitly marked**.

---

## 0. Prerequisites

We will use matrices to transform vector components, and two vector operations to describe geometric relationships.

The **dot product** measures alignment:

$$
\mathbf a\cdot\mathbf b
=\mathbf a^T\mathbf b
=\|\mathbf a\|\|\mathbf b\|\cos\theta.
$$

For nonzero vectors, its sign distinguishes acute from obtuse angles; perpendicular vectors have a dot product of zero.

The **cross product** in three dimensions gives a vector perpendicular to both inputs:

$$
\|\mathbf a\times\mathbf b\|
=\|\mathbf a\|\|\mathbf b\|\sin\theta,
\qquad
\mathbf a\times\mathbf b=-(\mathbf b\times\mathbf a).
$$

Its magnitude is the area of the parallelogram spanned by the inputs, and its direction follows the right-hand rule. The main special cases are:

| Relationship between nonzero vectors | Dot product                   | Cross product                                                 |
| ------------------------------------ | ----------------------------- | ------------------------------------------------------------- |
| Perpendicular                        | $0$                           | Perpendicular to both; magnitude $\|\mathbf a\|\|\mathbf b\|$ |
| Parallel, same direction             | $\|\mathbf a\|\|\mathbf b\|$  | $\mathbf0$                                                    |
| Parallel, opposite directions        | $-\|\mathbf a\|\|\mathbf b\|$ | $\mathbf0$                                                    |

A right-handed orthonormal frame satisfies $\mathbf e_x\times\mathbf e_y=\mathbf e_z$. Point your right thumb along a positive rotation axis; the curl of your fingers gives the positive rotation direction. Which axis points forward or upward is a separate convention, not something guaranteed by the words "right-handed."

---

## 1. Coordinate Frames and Poses

A **coordinate frame** consists of an origin and an ordered set of perpendicular unit axes. A pedestrian might have coordinates $(12,1,0)$ in a LiDAR frame and $(37,-8,0)$ in a world frame: each vector measures displacement from a different origin along a different set of axes.

A rigid body's **pose** combines:

- **position**: where the body-frame origin is located; and
- **orientation**: how the body-frame axes are aligned.

A **rigid** body keeps the distances between its points fixed. Its pose describes its position and orientation at one instant; a full dynamical state may also include velocity. A transform uses the pose to convert coordinates between frames.

It helps to start in two dimensions. A car's reference point needs two coordinates $(t_x,t_y)$, and its heading needs one angle $\theta$:

$$
\boxed{\text{2D pose}=(t_x,t_y,\theta).}
$$

These are three **degrees of freedom**, meaning three independently variable quantities. Attaching axes to the car creates its body frame: both the origin and axes move with the car. A point bolted to the car has constant body coordinates even when its world coordinates change.

![A car with an attached body frame, a fixed world frame, and the three planar pose parameters](poses-slide-13-body-frame.png)
_The primed axes move with the car; the world axes stay fixed._

In three dimensions, a free rigid body has six degrees of freedom: three translations and three rotations. Using roll, pitch, and yaw for orientation, we can collect them into

$$
\mathbf x
=
\begin{bmatrix}
t_x & t_y & t_z & \alpha & \beta & \gamma
\end{bmatrix}^{T},
$$

where $(\alpha,\beta,\gamma)$ denote roll, pitch, and yaw under the convention developed in Section 6. For computation, we will first represent orientation with a rotation matrix $\mathbf R$ and position with a translation vector $\mathbf t$.

![A simple sketch separating the three translational and three rotational degrees of freedom](pose-six-dof.svg)
_Three translational and three rotational degrees of freedom._

### 1.1 Notation

We will attach frame labels to each quantity. The left superscript in ${}^{B}\mathbf p$ identifies the frame in which the point's coordinates are expressed. A transform needs both a source and a destination:

$$
{}^{A}\mathbf T_B:\quad B\longrightarrow A.
$$

Read ${}^{A}\mathbf T_B$ as "map coordinates from frame $B$ into frame $A$." Rotation and translation use the same labels: ${}^{A}\mathbf R_B$ expresses the $B$ axes in frame $A$, while ${}^{A}\mathbf t_B$ locates the $B$ origin in frame $A$. Section 3 combines them into the matrix $\mathbf T$.

### 1.2 Building the transformation step by step

Given a point in body coordinates ${}^{B}\mathbf p$, we want to find its world coordinates ${}^{W}\mathbf p$. The construction is a head-to-tail sum of two displacements.

**Step 1: locate the body origin.** The translation

$$
{}^{W}\mathbf t_B
$$

runs from the world origin to the body origin, expressed along the world axes.

![The translation vector locating the body origin inside the world frame](step-1-translation.svg)
_The first displacement reaches the body origin._

**Step 2: describe the point relative to the body.** The vector ${}^{B}\mathbf p$ runs from the body origin to the point, but its components are measured along the body axes. Because the body axes are rotated relative to the world axes, we cannot add this vector directly to ${}^{W}\mathbf t_B$.

![A point represented as a displacement measured along the body-frame axes](step-2-local-point.svg)
_The second displacement is initially measured along the body axes._

**Step 3: rotate the local displacement into the world basis.** The rotation ${}^{W}\mathbf R_B$ converts components measured along the body axes into components measured along the world axes:

$$
{}^{W}\mathbf d
= {}^{W}\mathbf R_B\,{}^{B}\mathbf p.
$$

The displacement ${}^{W}\mathbf d$ now uses the same basis as the translation.

![The same displacement expressed first in the body basis and then in the world basis](step-3-change-basis.svg)
_Re-express the second displacement along the world axes._

**Step 4: add vectors that now share the same basis.** Both ${}^{W}\mathbf t_B$ and ${}^{W}\mathbf d$ are expressed in frame $W$, so they can be added:

$$
\begin{aligned}
{}^{W}\mathbf p
&= {}^{W}\mathbf t_B + {}^{W}\mathbf d \\
&= {}^{W}\mathbf t_B
 + {}^{W}\mathbf R_B\,{}^{B}\mathbf p.
\end{aligned}
$$

![A point expressed in a rotated body frame and transformed into the world frame](frames-transform.svg)

_The orange vector reaches the body origin, the red vector continues to the point, and their dashed purple sum gives its world position._

> [!note] Changing coordinates versus moving a point
> An **active rotation** moves a vector while its reference axes stay fixed. A **passive change of frame** describes the same geometric vector using different axes. Here, the frame-to-frame equations are coordinate conversions. Reversing the conversion requires the inverse rotation. Always identify the input and output frames before choosing a matrix.

---

## 2. Rigid Transforms

The construction above gives the **rigid transform**

$$
\boxed{{}^{W}\mathbf p
={}^{W}\mathbf R_B\,{}^{B}\mathbf p+{}^{W}\mathbf t_B.}
$$

To use it, we need to construct the rotation matrix and understand which properties make the mapping rigid.

### 2.1 Deriving the 2D rotation matrix

For a counterclockwise angle $\theta$, the body unit $x$-axis has world components $(\cos\theta,\sin\theta)^T$. The perpendicular body unit $y$-axis has world components $(-\sin\theta,\cos\theta)^T$. Put these vectors in the columns:

$$
{}^{W}\mathbf R_B=\mathbf R(\theta)
=\begin{bmatrix}
\cos\theta&-\sin\theta\\
\sin\theta&\cos\theta
\end{bmatrix}.
$$

Why columns? A point with body coordinates $(p_x,p_y)$ means $p_x$ steps along the body $x$-axis plus $p_y$ steps along the body $y$-axis. Matrix multiplication forms exactly that linear combination of columns. This also explains the columns of a 3D rotation matrix.

Suppose the body is rotated $90^\circ$ counterclockwise, its origin is at world position $(2,1)$, and a point has body coordinates $(2,1)$. Then

$$
\begin{aligned}
{}^{W}\mathbf p
&=\begin{bmatrix}0&-1\\1&0\end{bmatrix}
  \begin{bmatrix}2\\1\end{bmatrix}
  +\begin{bmatrix}2\\1\end{bmatrix}\\
&=\begin{bmatrix}-1\\2\end{bmatrix}
  +\begin{bmatrix}2\\1\end{bmatrix}
=\boxed{\begin{bmatrix}1\\3\end{bmatrix}}.
\end{aligned}
$$

### 2.2 What makes the transform rigid?

In either two or three dimensions, a proper rotation satisfies

$$
\mathbf R^T\mathbf R=\mathbf I,
\qquad
\det(\mathbf R)=1.
$$

The $(i,j)$ entry of $\mathbf R^T\mathbf R$ is the dot product of columns $i$ and $j$. A diagonal entry of $1$ gives unit length; an off-diagonal entry of $0$ gives perpendicularity. Together these properties are called **orthonormality**, and they imply $\mathbf R^{-1}=\mathbf R^T$. For the planar matrix, write $c=\cos\theta$ and $s=\sin\theta$:

$$
\mathbf R^T\mathbf R
=\begin{bmatrix}c&s\\-s&c\end{bmatrix}
 \begin{bmatrix}c&-s\\s&c\end{bmatrix}
=\begin{bmatrix}c^2+s^2&0\\0&c^2+s^2\end{bmatrix}
=\mathbf I,
\qquad \det\mathbf R=c^2+s^2=1.
$$

Now transform two points by $\mathbf a'=\mathbf R\mathbf a+\mathbf t$ and $\mathbf b'=\mathbf R\mathbf b+\mathbf t$. Their squared distance becomes

$$
\begin{aligned}
\|\mathbf a'-\mathbf b'\|_2^2
&=\|\mathbf R(\mathbf a-\mathbf b)\|_2^2\\
&=(\mathbf a-\mathbf b)^T\mathbf R^T\mathbf R(\mathbf a-\mathbf b)\\
&=\|\mathbf a-\mathbf b\|_2^2.
\end{aligned}
$$

Translation disappears in the difference, and rotation preserves the remaining norm. Inner products satisfy

$$
(\mathbf R\mathbf u)^T(\mathbf R\mathbf v)
=\mathbf u^T\mathbf R^T\mathbf R\mathbf v
=\mathbf u^T\mathbf v.
$$

Since both lengths and dot products remain unchanged, the cosine of the angle between any two nonzero displacement vectors remains unchanged too.

Orthogonal matrices can have determinant $+1$ or $-1$. The condition $\det(\mathbf R)=1$ rules out reflections, so handedness is preserved. For example, $\operatorname{diag}(-1,1)$ preserves planar distances but reflects across the $y$-axis; it is not a proper rotation.

![A square before and after a rigid transformation, with corresponding vertices and coordinate axes](poses-slide-19-rigid-properties.png)
_The square changes position and orientation while corresponding distances, angles, and handedness remain fixed._

---

## 3. Homogeneous Coordinates

**Homogeneous coordinates combine rotation and translation in one matrix**, allowing us to chain entire transforms by multiplication. The formula $\mathbf R\mathbf p+\mathbf t$ already works, but requires us to track the two operations separately.

The obstacle is that ordinary matrix multiplication cannot express a nonzero translation. Every matrix maps the zero vector to itself, $\mathbf A\mathbf0=\mathbf0$, but translation must move the origin to $\mathbf t$. This is why $\mathbf R\mathbf p+\mathbf t$ is affine rather than linear when $\mathbf t\ne\mathbf0$.

Appending a constant $1$ gives the matrix something to multiply the translation by. For example, translating a 2D point by $(5,2)$ becomes

$$
\begin{bmatrix}
1&0&5\\
0&1&2\\
0&0&1
\end{bmatrix}
\begin{bmatrix}x\\y\\1\end{bmatrix}
=
\begin{bmatrix}x+5\\y+2\\1\end{bmatrix}.
$$

The extra coordinate is not another physical dimension. It allows matrix multiplication to produce a constant offset. Adding rotation to the same construction gives the homogeneous point and transform:

$$
{}^{W}\hat{\mathbf p}
=
\begin{bmatrix}
{}^{W}\mathbf p\\1
\end{bmatrix},
\qquad
{}^{W}\mathbf T_B
=
\begin{bmatrix}
{}^{W}\mathbf R_B & {}^{W}\mathbf t_B\\
\mathbf 0^T & 1
\end{bmatrix}.
$$

Then a rigid transform becomes

$$
\boxed{
{}^{W}\hat{\mathbf p}
= {}^{W}\mathbf T_B\,{}^{B}\hat{\mathbf p}
}
$$

With this representation, a chain of frame conversions becomes a product of matrices. We can precompute that product and apply it to every point in a LiDAR scan; Section 5 works through the sensor-to-world example.

In 2D, the same construction uses a $3\times3$ matrix:

$$
\begin{bmatrix}{}^W p_x\\{}^W p_y\\1\end{bmatrix}
=
\begin{bmatrix}
\cos\theta&-\sin\theta&t_x\\
\sin\theta&\cos\theta&t_y\\
0&0&1
\end{bmatrix}
\begin{bmatrix}{}^B p_x\\{}^B p_y\\1\end{bmatrix}.
$$

In 3D, the $4\times4$ matrix belongs to the **special Euclidean group $SE(3)$**, the set of all proper 3D rigid transforms. **Euclidean** refers to preserving distances and angles, **special** excludes reflections, and **3** names the spatial dimension:

$$
SE(3)=\left\{
\begin{bmatrix}\mathbf R&\mathbf t\\\mathbf0^T&1\end{bmatrix}
\;\middle|\;
\mathbf R^T\mathbf R=\mathbf I,\quad
\det\mathbf R=1,\quad
\mathbf t\in\mathbb R^3
\right\}.
$$

Why call it a **group**? Its elements obey four rules under matrix multiplication:

- **Closure:** composing two rigid transforms gives another rigid transform, so $\mathbf T_1\mathbf T_2\in SE(3)$.
- **Associativity:** grouping does not change the result: $(\mathbf T_1\mathbf T_2)\mathbf T_3=\mathbf T_1(\mathbf T_2\mathbf T_3)$.
- **Identity:** $\mathbf I_4$ performs no rotation or translation.
- **Inverse:** every transform has an undo operation $\mathbf T^{-1}$, which also belongs to $SE(3)$.

Associativity lets us regroup a product, but does **not** let us swap its factors: generally, $\mathbf T_1\mathbf T_2\ne\mathbf T_2\mathbf T_1$.

The matrix has 12 nontrivial entries but only six degrees of freedom: the nine rotation entries are constrained, while the three translation entries are independent. Section 6 examines the rotation-only group $SO(3)$.

### 3.1 Points and directions are different

A point uses a final coordinate of $1$:

$$
\hat{\mathbf p}=\begin{bmatrix}\mathbf p\\1\end{bmatrix}.
$$

A direction uses a final coordinate of $0$:

$$
\hat{\mathbf v}=\begin{bmatrix}\mathbf v\\0\end{bmatrix}.
$$

Consequently,

$$
\begin{bmatrix}\mathbf R&\mathbf t\\\mathbf0^T&1\end{bmatrix}
\begin{bmatrix}\mathbf v\\0\end{bmatrix}
=
\begin{bmatrix}\mathbf R\mathbf v\\0\end{bmatrix}.
$$

Translation affects locations, not directions. For a rigid transform this distinction applies to surface normals, ray directions, axis vectors, and the components of a given velocity vector. Relating velocities measured by moving observers requires additional motion terms.

Subtracting two homogeneous points naturally produces a displacement with last coordinate zero:

$$
\begin{bmatrix}\mathbf a\\1\end{bmatrix}
-\begin{bmatrix}\mathbf b\\1\end{bmatrix}
=\begin{bmatrix}\mathbf a-\mathbf b\\0\end{bmatrix}.
$$

---

## 4. Inverting a Transform

To convert coordinates in the opposite direction, undo the translation first, then the rotation. Starting from ${}^{A}\mathbf p=\mathbf R\,{}^{B}\mathbf p+\mathbf t$ gives

$$
{}^{B}\mathbf p
=\mathbf R^T({}^{A}\mathbf p-\mathbf t)
=\mathbf R^T\,{}^{A}\mathbf p-\mathbf R^T\mathbf t.
$$

The inverse transform is therefore

$$
\boxed{
{}^{B}\mathbf T_A=({}^{A}\mathbf T_B)^{-1}
=\begin{bmatrix}
\mathbf R^T&-\mathbf R^T\mathbf t\\
\mathbf0^T&1
\end{bmatrix}.
}
$$

The inverse translation is $-\mathbf R^T\mathbf t$: negating the displacement reverses it, and multiplying by $\mathbf R^T$ expresses it in the reversed frame. This is why transposing the whole $4\times4$ matrix does not generally give its inverse.

Multiplication verifies the result:

$$
\begin{bmatrix}\mathbf R&\mathbf t\\\mathbf0^T&1\end{bmatrix}
\begin{bmatrix}\mathbf R^T&-\mathbf R^T\mathbf t\\\mathbf0^T&1\end{bmatrix}
=\begin{bmatrix}\mathbf I&\mathbf0\\\mathbf0^T&1\end{bmatrix}.
$$

For the example in Section 2.1, this recovers the original body coordinates:

$$
{}^{B}\mathbf p
=\begin{bmatrix}0&1\\-1&0\end{bmatrix}
\left(\begin{bmatrix}1\\3\end{bmatrix}
-\begin{bmatrix}2\\1\end{bmatrix}\right)
=\begin{bmatrix}2\\1\end{bmatrix}.
$$

---

## 5. Composing Coordinate Frames

Return to the LiDAR example. Perception supplies ${}^{S}\mathbf p$, calibration supplies ${}^{B}\mathbf T_S$, and localization supplies ${}^{W}\mathbf T_B$. These two transforms connect the measurement to the world.

![Sensor, body, and world frames connected by a chain of transforms](transform-chain.svg)
_The two frame mappings form one sensor-to-world path._

The desired world coordinate is

$$
\boxed{
{}^{W}\hat{\mathbf p}
= {}^{W}\mathbf T_B
  {}^{B}\mathbf T_S
  {}^{S}\hat{\mathbf p}
}
$$

Read the product from right to left: sensor coordinates become body coordinates, then world coordinates. The adjacent frame labels match, just as units must match in a dimensional calculation:

$$
{}^{W}\mathbf T_S
={}^W\mathbf T_B\,{}^B\mathbf T_S.
$$

Expanding the product shows how its rotation and translation combine:

$$
{}^{W}\mathbf T_S
=\begin{bmatrix}
{}^W\mathbf R_B\,{}^B\mathbf R_S
&{}^W\mathbf R_B\,{}^B\mathbf t_S+{}^W\mathbf t_B\\
\mathbf0^T&1
\end{bmatrix}.
$$

The sensor offset ${}^{B}\mathbf t_S$ is expressed in body coordinates. Rotate it into world coordinates before adding the body's world position ${}^{W}\mathbf t_B$. Simply adding the two translation vectors would mix bases.

### 5.1 A numerical 2D example

Suppose a sensor is mounted one meter ahead of a vehicle's body origin:

$$
{}^{B}\mathbf t_S=\begin{bmatrix}1\\0\end{bmatrix},
\qquad {}^{B}\mathbf R_S=\mathbf I.
$$

The vehicle is at $(10,5)$ in the world and rotated $90^\circ$ counterclockwise:

$$
{}^{W}\mathbf R_B=
\begin{bmatrix}0&-1\\1&0\end{bmatrix},
\qquad
{}^{W}\mathbf t_B=\begin{bmatrix}10\\5\end{bmatrix}.
$$

A pedestrian detected two meters along the sensor's $x$-axis has ${}^{S}\mathbf p=[2,0]^T$. First map sensor to body:

$$
{}^{B}\mathbf p=\begin{bmatrix}3\\0\end{bmatrix}.
$$

Then map body to world:

$$
{}^{W}\mathbf p
=
\begin{bmatrix}0&-1\\1&0\end{bmatrix}
\begin{bmatrix}3\\0\end{bmatrix}
+
\begin{bmatrix}10\\5\end{bmatrix}
=
\begin{bmatrix}10\\8\end{bmatrix}.
$$

The local phrase "three meters ahead" becomes the world location $(10,8)$ because the vehicle's forward axis points along world $+y$.

![Robot links and an autonomous vehicle with multiple camera and laser coordinate frames](poses-slide-43-sensor-frames.png)
_Real systems have many frames, but every measurement follows the same composition rule along a connected path._

For a rigidly mounted sensor, the calibration transform is usually fixed. Localization changes the body-to-world transform over time. On an articulated robot, joint motion also changes the transforms between links.

---

## 6. Rotation Matrices and $SO(3)$

So far, a matrix has represented orientation. We will now compare it with Euler angles, axis-angle, and quaternions. Each describes the same geometry but makes different calculations convenient.

We will follow **a positive $90^\circ$ turn about the $z$-axis** through all four representations. Here we use the active interpretation from Section 1.2: vectors rotate about the origin while the reference axes stay fixed.

### 6.1 Understanding what a rotation matrix stores

The column interpretation from Section 2.1 extends directly to 3D: **each column gives the destination of one unit axis**.

Write the standard unit vectors as $\mathbf e_x=(1,0,0)^T$, $\mathbf e_y=(0,1,0)^T$, and $\mathbf e_z=(0,0,1)^T$. Multiplication by each unit vector selects a column:

$$
\mathbf R=
\begin{bmatrix}
\vert&\vert&\vert\\
\mathbf R\mathbf e_x&\mathbf R\mathbf e_y&\mathbf R\mathbf e_z\\
\vert&\vert&\vert
\end{bmatrix}.
$$

For our $90^\circ$ turn about $z$,

$$
\mathbf e_x\mapsto\mathbf e_y,\qquad
\mathbf e_y\mapsto-\mathbf e_x,\qquad
\mathbf e_z\mapsto\mathbf e_z.
$$

Putting those destination vectors into columns gives

$$
\boxed{
\mathbf R_z(90^\circ)=
\begin{bmatrix}
0&-1&0\\
1&0&0\\
0&0&1
\end{bmatrix}.
}
$$

There is no need to memorize this particular matrix: following the three axes reconstructs it. To rotate any other vector, express that vector as a combination of the same axes. For example, $(2,1,3)^T$ means $2\mathbf e_x+\mathbf e_y+3\mathbf e_z$, so

$$
\mathbf R_z(90^\circ)
\begin{bmatrix}2\\1\\3\end{bmatrix}
=2\mathbf e_y-\mathbf e_x+3\mathbf e_z
=\begin{bmatrix}-1\\2\\3\end{bmatrix}.
$$

The orthonormality and handedness conditions established in Section 2.2 apply to these three columns. The set of all such matrices is called the **special orthogonal group $SO(3)$**:

$$
SO(3)=\left\{\mathbf R\in\mathbb R^{3\times3}
\mid \mathbf R^T\mathbf R=\mathbf I,\ \det\mathbf R=1\right\}.
$$

This is the rotation-only counterpart of $SE(3)$. Its identity is $\mathbf I_3$, its inverse is $\mathbf R^T$, and multiplication composes rotations.

Although a matrix stores nine entries, it has only **three independent degrees of freedom**. One way to see this geometrically is to choose its first unit axis $\mathbf r_1$: a direction on a sphere needs two parameters. The second axis $\mathbf r_2$ must be perpendicular to the first, leaving one angle around that first axis. Right-handedness then fixes the third axis as $\mathbf r_3=\mathbf r_1\times\mathbf r_2$.

Beyond the distance and angle preservation proved earlier, proper 3D rotations also preserve the cross product. Rotating two vectors first gives the same result as rotating their cross product:

$$
(\mathbf R\mathbf a)\times(\mathbf R\mathbf b)
=\mathbf R(\mathbf a\times\mathbf b).
$$

For example, our quarter-turn sends $\mathbf e_x$ to $\mathbf e_y$ and $\mathbf e_y$ to $-\mathbf e_x$. Their new cross product is still $\mathbf e_z$, just as rotating the original cross product $\mathbf e_x\times\mathbf e_y=\mathbf e_z$ leaves it unchanged.

For a general proof, recall that $(\mathbf a\times\mathbf b)\cdot\mathbf c$ is the signed volume spanned by three vectors. A proper rotation preserves this volume because $\det\mathbf R=1$. Therefore, for every $\mathbf c$,

$$
\begin{aligned}
\big((\mathbf R\mathbf a)\times(\mathbf R\mathbf b)\big)\cdot\mathbf R\mathbf c
&=(\mathbf a\times\mathbf b)\cdot\mathbf c\\
&=\big(\mathbf R(\mathbf a\times\mathbf b)\big)\cdot\mathbf R\mathbf c.
\end{aligned}
$$

The two candidate vectors have the same dot product with every possible direction, so they must be equal.

**Applications also need to calculate new orientations.** Three tasks motivate the choice of representation:

- **Composition** connects a camera orientation to the world through the robot: ${}^{W}\mathbf R_C={}^W\mathbf R_B\,{}^B\mathbf R_C$. This is the rotation part of the frame chain in Section 5, handled naturally by matrix multiplication.
- **Interpolation** fills in a smooth motion. If a camera must turn $90^\circ$ left in one second, a constant-speed turn places it at $45^\circ$ halfway through. An animation or controller needs these intermediate orientations.
- **Optimization** improves an orientation estimate. A robot starts with a guess, compares its predicted camera observations with the actual measurements, and adjusts the guess to reduce the error. Each update must remain a valid rotation.

The last two tasks require more care than ordinary arithmetic on matrix entries. For example, averaging the matrices for no rotation and a $180^\circ$ turn about $z$ gives

$$
\frac12\left(
\begin{bmatrix}1&0&0\\0&1&0\\0&0&1\end{bmatrix}
+
\begin{bmatrix}-1&0&0\\0&-1&0\\0&0&1\end{bmatrix}
\right)
=
\begin{bmatrix}0&0&0\\0&0&0\\0&0&1\end{bmatrix}.
$$

The average collapses $\mathbf e_x$ to zero rather than giving an intermediate orientation. Arbitrary optimization updates can likewise destroy perpendicularity, which normalizing each column alone cannot restore. Quaternions and rotation vectors provide convenient alternatives for these tasks; applications that only apply or compose rotations can continue using matrices.

### 6.2 Euler angles: describing a turn as three steps

Euler angles describe a rotation through a sequence of simpler turns. For roll, pitch, and yaw, the three parameters are

| Parameter     | Axis | Physical intuition when $x$ is forward, $y$ is left, and $z$ is up |
| ------------- | ---- | ------------------------------------------------------------------ |
| Roll $\alpha$ | $x$  | Banking from side to side                                          |
| Pitch $\beta$ | $y$  | Tilting the nose up or down                                        |
| Yaw $\gamma$  | $z$  | Turning left or right                                              |

These descriptions identify the type of motion; the right-hand rule determines its sign. Under the convention used here,

$$
\boxed{\mathbf R=\mathbf R_z(\gamma)\mathbf R_y(\beta)\mathbf R_x(\alpha).}
$$

Read this equation as a recipe applied to a column vector:

$$
\begin{aligned}
\mathbf p_1 &= \mathbf R_x(\alpha)\mathbf p,\\
\mathbf p_2 &= \mathbf R_y(\beta)\mathbf p_1,\\
\mathbf p' &= \mathbf R_z(\gamma)\mathbf p_2.
\end{aligned}
$$

Substituting the intermediate vectors produces the matrix product above. **The rightmost matrix acts first because it is closest to the input vector.**

Each elemental matrix leaves one coordinate unchanged and performs a 2D rotation in the perpendicular plane:

$$
\mathbf R_x(\alpha)=
\begin{bmatrix}
1&0&0\\
0&\cos\alpha&-\sin\alpha\\
0&\sin\alpha&\cos\alpha
\end{bmatrix}.
$$

Here the $x$ component stays fixed while the $y,z$ components rotate. Similarly,

$$
\mathbf R_y(\beta)=
\begin{bmatrix}
\cos\beta&0&\sin\beta\\
0&1&0\\
-\sin\beta&0&\cos\beta
\end{bmatrix},
$$

$$
\mathbf R_z(\gamma)=
\begin{bmatrix}
\cos\gamma&-\sin\gamma&0\\
\sin\gamma&\cos\gamma&0\\
0&0&1
\end{bmatrix}.
$$

Our running example has $(\alpha,\beta,\gamma)=(0,0,90^\circ)$. The first two matrices are identities, leaving exactly the quarter-turn matrix from Section 6.1.

To see the effect of order, follow $\mathbf e_x$ through two sequences:

$$
\begin{aligned}
\mathbf e_x
&\xrightarrow{\;R_x(90^\circ)\;}\mathbf e_x
\xrightarrow{\;R_z(90^\circ)\;}\mathbf e_y,\\
\mathbf e_x
&\xrightarrow{\;R_z(90^\circ)\;}\mathbf e_y
\xrightarrow{\;R_x(90^\circ)\;}\mathbf e_z.
\end{aligned}
$$

A turn about $x$ does nothing to a vector already on the $x$-axis. After a turn about $z$, however, that vector points along $y$, so a subsequent turn about $x$ does affect it. Thus $\mathbf R_z\mathbf R_x$ and $\mathbf R_x\mathbf R_z$ are generally different.

The axes used at each step must also be specified. **Extrinsic** rotations use fixed reference axes; **intrinsic** rotations use axes that move with the body. For example, "turn about the room's vertical axis" names a fixed axis, whereas "turn about the object's current forward axis" names a moving one. The product above is fixed-axis $x$ then $y$ then $z$; it is equivalently moving-axis $z$ then the new $y$ then the new $x$, starting from aligned frames.

There are twelve standard three-axis sequences: six with distinct axes ($XYZ$, $XZY$, $YXZ$, $YZX$, $ZXY$, $ZYX$) and six that repeat the first axis ($XYX$, $XZX$, $YXY$, $YZY$, $ZXZ$, $ZYZ$). The distinct-axis cases are also called _Tait–Bryan angles_. You do not need to memorize the list to use this section; you do need to record the chosen sequence, fixed or moving axes, and angle units.

### 6.3 Gimbal lock: when two angle controls do the same job

Euler angles use three numbers, but at certain orientations those three controls stop producing three independent local changes. A physical gimbal makes this visible: it holds an object in nested rotating rings, and two of the rings' axes can line up. Once aligned, turning either ring rotates about the same line.

![Two gimbal configurations showing separated rotation axes and an aligned-axis singular configuration](poses-slide-50-gimbal-lock.png)
_The configuration on the right has two aligned gimbal axes. The ring labels belong to this particular mechanism; the roll–yaw calculation below uses our stated Euler convention._

For our convention, this happens when pitch is $\beta=\pi/2$, or $90^\circ$. At that value, $\cos\beta=0$ and $\sin\beta=1$. Substituting them into $\mathbf R_z(\gamma)\mathbf R_y(\beta)\mathbf R_x(\alpha)$ gives

$$
\mathbf R=
\begin{bmatrix}
0&\sin(\alpha-\gamma)&\cos(\alpha-\gamma)\\
0&\cos(\alpha-\gamma)&-\sin(\alpha-\gamma)\\
-1&0&0
\end{bmatrix}.
$$

**Roll and yaw appear only as $\alpha-\gamma$**, so they cannot be recovered separately. For example, both triples

$$
(\alpha,\beta,\gamma)=(0^\circ,90^\circ,0^\circ)
$$

and

$$
(\alpha,\beta,\gamma)=(30^\circ,90^\circ,30^\circ)
$$

produce the same rotation because both have $\alpha-\gamma=0$. Increasing roll and yaw together by any equal amount leaves the orientation unchanged. Increasing roll alone has the same effect as decreasing yaw alone.

Near this singularity, even a small orientation change can require large changes in the reported angles, making estimation and control unstable. The same convention becomes singular at pitch $-90^\circ$, with a different roll–yaw coupling.

> [!important] The object does not lose the ability to rotate
> A freely moving rigid body still has three rotational degrees of freedom. Its **Euler-angle coordinates** cannot describe three independent infinitesimal changes at gimbal lock. A quaternion or rotation matrix can represent that same orientation without this Euler singularity.

### 6.4 Axis-angle: one axis and one amount of turning

**Axis-angle represents the entire rotation as one turn about one axis.** Every proper 3D rotation admits this description, with the axis passing through the origin. Translation remains separate.

Let $\mathbf u$ be a unit vector along the axis, and let $\theta$ be the right-handed rotation angle. For our running quarter-turn,

$$
\mathbf u=\begin{bmatrix}0\\0\\1\end{bmatrix},
\qquad \theta=\frac{\pi}{2}.
$$

The axis stores three components, but the constraint $\|\mathbf u\|=1$ leaves only two independent parameters. Adding the angle gives three degrees of freedom overall.

The **rotation vector**, also called the Euler vector, combines the axis and angle:

$$
\boxed{\boldsymbol\omega=\theta\mathbf u.}
$$

Its direction gives the axis and its length gives the angle. For our example, $\boldsymbol\omega=(0,0,\pi/2)^T$. When $\boldsymbol\omega\ne\mathbf0$, recover the angle as $\theta=\|\boldsymbol\omega\|$ and the axis as $\mathbf u=\boldsymbol\omega/\|\boldsymbol\omega\|$. The zero vector represents no rotation.

A rotation vector is **not** a triple of roll, pitch, and yaw angles. Its components encode one axis scaled by one angle, not three sequential turns.

![A unit rotation axis, a circular rotation arrow, and a rotation vector aligned with the axis](poses-slide-52-axis-angle.png)
_The axis sets the direction of the turn. The figure labels its angle $\psi$ or $\Psi$; our equations use $\theta$._

To rotate a vector about this axis, split it into parallel and perpendicular components:

$$
\mathbf v_\parallel=(\mathbf u^T\mathbf v)\mathbf u,
\qquad
\mathbf v_\perp=\mathbf v-\mathbf v_\parallel.
$$

The dot product measures the component along the unit axis; multiplying by $\mathbf u$ reconstructs that parallel vector. The remainder lies in the plane of rotation.

The parallel part stays fixed. The perpendicular part turns in a 2D plane. Within that plane, $\mathbf u\times\mathbf v_\perp$ is a $90^\circ$ turn of $\mathbf v_\perp$ with the same length. Just as a planar rotation combines cosine and sine components,

$$
\boxed{
\mathbf R\mathbf v
=\mathbf v_\parallel
+\cos\theta\,\mathbf v_\perp
+\sin\theta\,(\mathbf u\times\mathbf v).
}
$$

We used $\mathbf u\times\mathbf v$ because the parallel component contributes zero to that cross product.

For a concrete example, rotate $\mathbf v=(1,0,2)^T$ by $90^\circ$ about $z$. Its parallel part is $(0,0,2)^T$, its perpendicular part is $(1,0,0)^T$, and $\mathbf u\times\mathbf v=(0,1,0)^T$. Since $\cos90^\circ=0$ and $\sin90^\circ=1$,

$$
\mathbf R\mathbf v
=\begin{bmatrix}0\\0\\2\end{bmatrix}
+\begin{bmatrix}0\\1\\0\end{bmatrix}
=\begin{bmatrix}0\\1\\2\end{bmatrix}.
$$

The height stays fixed while the horizontal part turns. This formula already lets us rotate vectors directly, without first constructing a matrix.

**Rodrigues' formula packages this geometry into a matrix.** Define

$$
[\mathbf u]_\times=
\begin{bmatrix}
0&-u_z&u_y\\
u_z&0&-u_x\\
-u_y&u_x&0
\end{bmatrix},
\qquad
[\mathbf u]_\times\mathbf v=\mathbf u\times\mathbf v.
$$

The bracket notation is simply a matrix implementation of "take the cross product with $\mathbf u$." Substituting the parallel and perpendicular components into the vector formula gives

$$
\mathbf R
=\cos\theta\,\mathbf I
+(1-\cos\theta)\mathbf u\mathbf u^T
+\sin\theta[\mathbf u]_\times.
$$

Here $\mathbf u\mathbf u^T$ projects onto the rotation axis. The identity

$$
\mathbf u\times(\mathbf u\times\mathbf v)
=\mathbf u(\mathbf u^T\mathbf v)-\mathbf v
$$

for a unit axis implies $[\mathbf u]_\times^2=\mathbf u\mathbf u^T-\mathbf I$. Using it yields the usual form:

$$
\boxed{
\mathbf R=\mathbf I+\sin\theta[\mathbf u]_\times
+(1-\cos\theta)[\mathbf u]_\times^2.
}
$$

**Axis-angle is not unique.** Reversing both the axis and angle sign, or adding $2\pi$ to the angle, leaves the rotation unchanged. At zero angle any axis works; at angle $\pi$, either axis direction works.

A common rotation-vector convention restricts $\|\boldsymbol\omega\|\le\pi$, but opposite points on the boundary represent the same rotation. This creates a numerical seam. Consider two turns about $z$, expressed here in degrees for readability:

$$
\boldsymbol\omega_1=(0,0,179^\circ)^T,
\qquad
\boldsymbol\omega_2=(0,0,-179^\circ)^T.
$$

Both are nearly a half-turn, so their actual orientations are only $2^\circ$ apart. Their coordinate difference is nevertheless $358^\circ$. Averaging the vectors gives zero rotation, which is nowhere near the short-path midpoint at $180^\circ$.

Rotation vectors are useful for small local changes. For small $\theta$ in radians, $\sin\theta\approx\theta$ and $1-\cos\theta\approx0$ to first order, so Rodrigues' formula becomes $\mathbf R\approx\mathbf I+[\boldsymbol\omega]_\times$. This approximation helps explain their role in optimization, but it does not make global vector averaging valid.

### 6.5 Unit quaternions: encoding axis-angle in four numbers

**A unit quaternion encodes axis-angle using four real numbers.** Starting from that connection lets us understand its geometry before introducing its multiplication rule.

Using scalar-first order, write

$$
\mathbf q=(w,\mathbf v)=(w,x,y,z),
\qquad
\mathbf v=\begin{bmatrix}x\\y\\z\end{bmatrix}.
$$

The three numbers in $\mathbf v$ point along the rotation axis, but their length is generally not one. Given a unit axis $\mathbf u$ and angle $\theta$, set

$$
\boxed{
\mathbf q=\left(\cos\frac{\theta}{2},\mathbf u\sin\frac{\theta}{2}\right).
}
$$

The scalar part stores the cosine of the half-angle; the vector part stores the axis scaled by the sine of the half-angle. These four numbers satisfy

$$
w^2+x^2+y^2+z^2
=\cos^2\frac{\theta}{2}+\sin^2\frac{\theta}{2}=1.
$$

That is what **unit** means. Four stored numbers minus one constraint leaves three degrees of freedom; there is no fourth physical direction of rotation.

For our $90^\circ$ turn about $z$,

$$
\mathbf q=\left(\frac{\sqrt2}{2},0,0,\frac{\sqrt2}{2}\right).
$$

The identity rotation has quaternion $(1,0,0,0)$, and a $180^\circ$ turn about $z$ has quaternion $(0,0,0,1)$. These examples make clear that the vector part is not an ordinary position vector. Some libraries store $(x,y,z,w)$ instead, so component order must be checked.

To compose rotations, quaternions use a multiplication rule called the **Hamilton product**. For $\mathbf q_1=(w_1,\mathbf v_1)$ and $\mathbf q_2=(w_2,\mathbf v_2)$, write the result as $(w_{12},\mathbf v_{12})$, where

$$
\begin{aligned}
w_{12}&=w_1w_2-\mathbf v_1\cdot\mathbf v_2,\\
\mathbf v_{12}&=w_1\mathbf v_2+w_2\mathbf v_1+\mathbf v_1\times\mathbf v_2.
\end{aligned}
$$

We denote the result by $\mathbf q_1\otimes\mathbf q_2$. This is not componentwise multiplication: the dot and cross products couple the inputs. Swapping them changes the cross-product sign, making composition order-dependent.

The notation $\mathbf q=w+xi+yj+zk$ describes the same algebra using imaginary units satisfying $i^2=j^2=k^2=ijk=-1$. For instance, $ij=k$ while $ji=-k$. You can compute rotations entirely with the four real components and the Hamilton formula above.

To reverse a rotation, use the quaternion **conjugate**, which negates the vector part:

$$
\mathbf q^*=(w,-\mathbf v).
$$

For a unit quaternion this is the inverse, $\mathbf q^{-1}=\mathbf q^*$. Geometrically, the sign change reverses the angle about the same axis. For a general nonzero quaternion, the inverse is instead $\mathbf q^*/\|\mathbf q\|^2$; the unit-norm assumption matters.

To apply the rotation to a 3D vector, package it as a quaternion with scalar part zero, $\mathbf p_q=(0,\mathbf p)$, and compute

$$
\boxed{\mathbf p'_q=\mathbf q\otimes\mathbf p_q\otimes\mathbf q^{-1}.}
$$

The output has zero scalar part and the rotated vector as its remaining components. Both multiplications are needed.

This also explains the half-angle. Take a rotation about $z$, write $\mathbf q=(c,0,0,s)$ with $c=\cos(\theta/2)$ and $s=\sin(\theta/2)$, and rotate $\mathbf e_x$. The two products give

$$
\mathbf q\otimes(0,1,0,0)\otimes\mathbf q^{-1}
=(0,c^2-s^2,2cs,0)
=(0,\cos\theta,\sin\theta,0).
$$

The vector part has turned through the full angle $\theta$. At $90^\circ$, it is exactly $(0,1,0)$, agreeing with the matrix and axis-angle examples.

Under this convention, applying $\mathbf q_1$ first and $\mathbf q_2$ second uses $\mathbf q_2\otimes\mathbf q_1$. For example, composing two $90^\circ$ turns about $z$ multiplies $(\sqrt2/2,0,0,\sqrt2/2)$ by itself, giving $(0,0,0,1)$: a $180^\circ$ turn.

The two-sided formula also explains why **$\mathbf q$ and $-\mathbf q$ represent the same rotation**. Negating the quaternion negates its inverse as well, so the two signs cancel:

$$
(-\mathbf q)\otimes\mathbf p_q\otimes(-\mathbf q)^{-1}
=\mathbf q\otimes\mathbf p_q\otimes\mathbf q^{-1}.
$$

Every orientation therefore has two opposite quaternion representatives, a property called a **double cover**. An apparent jump from $\mathbf q$ to $-\mathbf q$ need not mean any physical motion. When processing a sequence, choose signs consistently, and renormalize quaternions if numerical error causes their lengths to drift.

### 6.6 Interpolating orientations with SLERP

We can now solve the camera-interpolation problem from Section 6.1. Unit quaternions lie on a sphere in four-dimensional space. A straight line between two of them generally passes inside that sphere; **spherical linear interpolation (SLERP)** follows an arc on its surface instead. Let $s\in[0,1]$ measure progress from the starting orientation to the ending orientation.

Start with unit quaternions $\mathbf q_0$ and $\mathbf q_1$. If $\mathbf q_0\cdot\mathbf q_1<0$, replace $\mathbf q_1$ with $-\mathbf q_1$. This leaves its physical orientation unchanged while selecting a shorter arc. Let

$$
\Omega=\arccos(\mathbf q_0\cdot\mathbf q_1).
$$

Here $\Omega$ is the angle between the quaternions in four-dimensional space. After this sign choice, the corresponding shortest physical rotation angle is $2\Omega$. For $0\le s\le1$,

$$
\boxed{
\mathbf q(s)=
\frac{\sin((1-s)\Omega)}{\sin\Omega}\mathbf q_0
+\frac{\sin(s\Omega)}{\sin\Omega}\mathbf q_1.
}
$$

The sine weights replace the straight-line weights $1-s$ and $s$. They keep the result on the unit sphere and make angular progress proportional to $s$. At $s=0$ the second weight vanishes and the first is one; at $s=1$ the reverse happens.

For example, interpolate from no rotation, $\mathbf q_0=(1,0,0,0)$, to our $90^\circ$ turn, $\mathbf q_1=(\sqrt2/2,0,0,\sqrt2/2)$. Their dot product is $\sqrt2/2$, so $\Omega=45^\circ$. Halfway through,

$$
\mathbf q(1/2)=\left(\cos22.5^\circ,0,0,\sin22.5^\circ\right),
$$

which represents a **$45^\circ$ physical turn** about $z$. If $s$ advances uniformly with time, the camera turns at constant angular speed along this path.

For implementation, clamp the dot product to $[-1,1]$ before applying $\arccos$. When $\Omega$ is very small, dividing by $\sin\Omega$ is numerically awkward. A useful approximation is normalized linear interpolation:

$$
\mathbf q(s)\approx
\frac{(1-s)\mathbf q_0+s\mathbf q_1}
{\|(1-s)\mathbf q_0+s\mathbf q_1\|}.
$$

Use the same sign choice first. This approximation preserves unit norm, though its angular speed is not exactly constant in general.

### 6.7 Choosing a representation

Choose a representation according to what you need to do:

| Representation  | Stored values | Useful when                                              | What to watch for                                               |
| --------------- | ------------: | -------------------------------------------------------- | --------------------------------------------------------------- |
| Rotation matrix |             9 | Rotating point clouds and composing frame mappings       | Columns must remain orthonormal and right-handed                |
| Euler angles    |             3 | Displaying understandable roll, pitch, and yaw           | Axis conventions, order, and gimbal lock                        |
| Axis-angle      |             4 | Explaining a turn geometrically                          | Nonunique axes at zero angle and sign ambiguity at $\pi$        |
| Rotation vector |             3 | Expressing small rotation updates                        | Global subtraction and averaging can cross a boundary           |
| Unit quaternion |             4 | Storing orientations, composing turns, and interpolating | Unit norm, component order, and $\mathbf q$ versus $-\mathbf q$ |

A pipeline can store a pose with a quaternion and translation, convert it to a matrix to transform a point cloud, and display Euler angles to a person. These conversions change the representation while preserving the orientation.

---

## 7. Uniform Sampling on Curved Spaces

The geometry that complicates interpolation also matters when generating random directions or orientations. Uniformly sampling coordinates does not necessarily produce a uniform geometric distribution: equal coordinate intervals may cover different amounts of area or volume.

### 7.1 Uniform directions on the sphere

With azimuth $\phi\in[0,2\pi)$ and polar angle $\theta\in[0,\pi]$, a unit direction is

$$
\mathbf d=
\begin{bmatrix}
\sin\theta\cos\phi\\
\sin\theta\sin\phi\\
\cos\theta
\end{bmatrix}.
$$

Here $\theta$ is the **polar angle**, measured down from the positive $z$-axis; it differs from geographic latitude.

The sphere's area element is

$$
dA=\sin\theta\,d\theta\,d\phi.
$$

Near a pole, a fixed angular rectangle covers less area than near the equator. Uniformly sampling both angles therefore produces too many samples near the poles per unit area.

Instead, sample $z\sim U(-1,1)$ and $\phi\sim U(0,2\pi)$ independently, then set

$$
\boxed{
\mathbf d=
\begin{bmatrix}
\sqrt{1-z^2}\cos\phi\\
\sqrt{1-z^2}\sin\phi\\
z
\end{bmatrix}.
}
$$

Since $z=\cos\theta$, the area element becomes $|dz|\,d\phi$: equal-height strips of a sphere have equal area. Another valid method is to normalize a vector of three independent standard normal samples: its rotationally symmetric distribution has no preferred direction.

### 7.2 Uniform rotations in $SO(3)$

A random direction is not a full random orientation: after choosing where one axis points, rotation about that axis is still free. Nor does sampling roll, pitch, and yaw independently and uniformly yield uniform orientations.

For $\mathbf R=\mathbf R_z(\gamma)\mathbf R_y(\beta)\mathbf R_x(\alpha)$ with $\beta\in[-\pi/2,\pi/2]$, the uniform rotation measure is proportional to

$$
\cos\beta\,d\alpha\,d\beta\,d\gamma.
$$

One valid sampler draws roll and yaw uniformly over a full turn, samples $s\sim U(-1,1)$, and sets $\beta=\arcsin s$.

An especially simple alternative uses unit quaternions:

1. Draw four independent standard normal samples $g_0,g_1,g_2,g_3$.
2. Form $\mathbf g=(g_0,g_1,g_2,g_3)$.
3. Normalize: $\mathbf q=\mathbf g/\|\mathbf g\|$.
4. Interpret $\mathbf q$ as a unit quaternion.

The normalized Gaussian is uniform on the three-dimensional unit sphere $S^3$ in four-dimensional space. Its quaternion-to-rotation mapping produces the uniform, or **Haar**, distribution on $SO(3)$. The two-to-one mapping is harmless because every rotation has exactly the pair $\mathbf q,-\mathbf q$.

> [!note] Normalize Gaussian samples, not uniform cube samples
> Normalizing four samples from $U(-1,1)$ does not give the same distribution. A Gaussian is rotationally symmetric; a cube is not. The same distinction applies when sampling directions on $S^2$.

---

## 8. Implementation Pattern

The geometry translates directly into NumPy. In variable names, `A_T_B` means the transform from frame B to frame A; the functions below assume valid rigid-transform inputs.

```python
import numpy as np


def make_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Construct destination_T_source from destination_R_source and destination_t_source."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def invert_transform(A_T_B: np.ndarray) -> np.ndarray:
    """Return B_T_A."""
    R = A_T_B[:3, :3]
    t = A_T_B[:3, 3]

    B_T_A = np.eye(4)
    B_T_A[:3, :3] = R.T
    B_T_A[:3, 3] = -R.T @ t
    return B_T_A


def transform_point(A_T_B: np.ndarray, B_p: np.ndarray) -> np.ndarray:
    """Map a 3D point from frame B to frame A."""
    B_p_h = np.append(B_p, 1.0)
    return (A_T_B @ B_p_h)[:3]


def is_rotation_matrix(R: np.ndarray, atol: float = 1e-7) -> bool:
    return (
        np.allclose(R.T @ R, np.eye(3), atol=atol, rtol=0.0)
        and np.isclose(np.linalg.det(R), 1.0, atol=atol, rtol=0.0)
    )
```

Reusing the sensor setup from Section 5.1 in 3D lets us check composition, inversion, point-versus-direction behavior, and distance preservation:

```python
W_T_B = make_transform(
    np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]),
    np.array([10., 5., 0.]),
)
B_T_S = make_transform(np.eye(3), np.array([1., 0., 0.]))
S_p = np.array([2., 0., 0.])
W_T_S = W_T_B @ B_T_S
W_p = transform_point(W_T_S, S_p)

assert np.allclose(W_p, [10., 8., 0.])
assert np.allclose(W_T_S @ invert_transform(W_T_S), np.eye(4))
assert np.allclose(transform_point(invert_transform(W_T_S), W_p), S_p)
assert is_rotation_matrix(W_T_S[:3, :3])

# The last homogeneous coordinate controls whether translation applies.
assert np.allclose(W_T_B @ [1., 0., 0., 0.], [0., 1., 0., 0.])
assert np.allclose(W_T_B @ [1., 0., 0., 1.], [10., 6., 0., 1.])

S_p2 = np.array([-1., 2., 4.])
assert np.isclose(
    np.linalg.norm(transform_point(W_T_S, S_p) - transform_point(W_T_S, S_p2)),
    np.linalg.norm(S_p - S_p2),
)
```

One normalization routine implements both Gaussian samplers from Section 7. The four-dimensional result is interpreted in scalar-first `(w, x, y, z)` order:

```python
rng = np.random.default_rng(42)


def sample_unit_vectors(rng, count: int, dimension: int) -> np.ndarray:
    samples = rng.normal(size=(count, dimension))
    norms = np.linalg.norm(samples, axis=1, keepdims=True)
    # Exact zeros have probability zero in the ideal distribution;
    # resample them defensively for finite-precision computation.
    zero = norms[:, 0] == 0
    while np.any(zero):
        samples[zero] = rng.normal(size=(np.count_nonzero(zero), dimension))
        norms = np.linalg.norm(samples, axis=1, keepdims=True)
        zero = norms[:, 0] == 0
    return samples / norms


directions = sample_unit_vectors(rng, count=1000, dimension=3)
quaternions_wxyz = sample_unit_vectors(rng, count=1000, dimension=4)
```

> [!warning] Library conventions differ
> Before combining libraries, check axis handedness, quaternion component order (`wxyz` versus `xyzw`), angle units, Euler order, whether vectors are rows or columns, and whether a matrix maps source-to-destination or destination-to-source. Shape compatibility alone does not prove semantic compatibility.

---

## 9. Summary

Keep frame labels attached to every quantity, then choose the operation that matches the task:

- **Convert a point:** ${}^{A}\mathbf p={}^A\mathbf R_B\,{}^B\mathbf p+{}^A\mathbf t_B$.
- **Chain frames:** ${}^{A}\mathbf T_C={}^A\mathbf T_B\,{}^B\mathbf T_C$; adjacent frame labels must match.
- **Reverse a mapping:** use $\mathbf R^T$ and $-\mathbf R^T\mathbf t$.
- **Interpolate, estimate, or sample orientations:** respect the constraints of the chosen representation rather than treating its entries as unconstrained coordinates.

---

## References

- [3D Rotations - Robotic Systems](http://motion.cs.illinois.edu/RoboticSystems/3DRotations.html).
- [Visualizing Quaternions](https://eater.net/quaternions/).

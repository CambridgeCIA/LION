# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
from dataclasses import dataclass


@dataclass
class A:
    """A simple dataclass with one integer field `x` initialized to 0.

    Parameters
    ----------
    x : int
        An integer field initialized to 0 by default.
    """

    x: int = 0


# class B should still has @dataclass even if it inherits from A, see below examples comparing with and without @dataclass (C0 and C1)
@dataclass
class B(A):
    """A dataclass that inherits from A and adds an additional integer field `y` initialized to 1.

    Parameters
    ----------
    y : int
        An integer field initialized to 1 by default.
    """

    y: int = 1


b = B()
print(f"b.x = {b.x}")  # Output: 0

# %%
# Test that we can add new fields to an instance of B even though it is a dataclass, and that the new field is not shared across instances of B
b.z = 2
print(f"b.z = {b.z}")  # Output: 2

b2 = B()
print(
    f"b2 has field z: {hasattr(b2, 'z')}"
)  # Output: False, b2 does not have the field z, so it is not shared across instances of B


# %%


class C0(B):
    """A class that inherits from B but does not have the @dataclass decorator. It attempts to override the field `x` from A, but since it doesn't have the @dataclass decorator, it doesn't actually override it for instances of C0."""

    x: int = (
        3  # Without @dataclass, this is the class variable, not instance variable, so it doesn't override A.x
    )


c0 = C0()
print(
    f"c0.x = {c0.x}"
)  # Output: 0, same as A.x, not 3, because C0 doesn't have decorator @dataclass
print(f"C0.x = {C0.x}")


@dataclass
class C1(B):
    """A dataclass that inherits from B and has the @dataclass decorator. It overrides the field `x` from A, and since it has the @dataclass decorator, it successfully overrides it for instances of C1.

    Parameters
    ----------
    x : int
        An integer field initialized to 3 by default, which overrides the field `x` from A.
    """

    x: int = (
        3  # With @dataclass, this is the instance variable, so it overrides A.x for instances of C1
    )


c1 = C1()
print(
    f"c1.x = {c1.x}"
)  # Output: 3, because C1 has decorator @dataclass, so it overrides A.x


# %%
class D:
    def __init__(self, x=0):
        self.x = x


d = D()
d.z = 2
print(f"d.z = {d.z}")  # Output: 2

# %%
from abc import ABC


class E(ABC):
    def some_method(self):
        """A method that should be implemented by subclasses of E."""


class F(E):
    """A class that inherits from E but does not implement the some_method,
    so it is still an abstract class and cannot be instantiated.
    """


f = (
    F()
)  # This will raise a TypeError because F is still an abstract class due to not implementing some_method
print(f"f = {f}")
print(
    f"f.some_method() = {f.some_method()}"
)  # This will not be reached due to the error above

# %%

# Test tyro.cli with to see if it also shows help for inherited fields from parent classes


def show_help():
    # b_from_cli = tyro.cli(B)
    # print(f"b_from_cli = {b_from_cli}")
    # c0_from_cli = tyro.cli(C0)
    # print(f"c0_from_cli = {c0_from_cli}")
    # c1_from_cli = tyro.cli(C1)
    # print(f"c1_from_cli = {c1_from_cli}")
    pass


show_help()

# %%
from LION.experiments.ct_experiments import ExtremeLowDoseCTRecon

experiment = ExtremeLowDoseCTRecon()
print(experiment)

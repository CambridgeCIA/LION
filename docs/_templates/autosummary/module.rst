{{ fullname | escape | underline }}

.. automodule:: {{ fullname }}
   :no-members:

{% block modules %}
{% if modules %}
.. rubric:: Modules

.. autosummary::
   :toctree:
   :recursive:
{% for item in modules %}
   {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}

{% block classes %}
{% if classes %}
.. rubric:: Classes

.. autosummary::
{% for item in classes %}
   {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}

{% block functions %}
{% if functions %}
.. rubric:: Functions

.. autosummary::
{% for item in functions %}
   {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}

{% if classes or functions %}
----

{% for item in classes %}
.. autoclass:: {{ item }}
   :members:
   :show-inheritance:

{% endfor %}
{% for item in functions %}
.. autofunction:: {{ item }}

{% endfor %}
{% endif %}

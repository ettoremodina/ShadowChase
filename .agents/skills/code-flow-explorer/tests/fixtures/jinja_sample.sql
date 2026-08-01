{% macro choose_query(items) %}
{% set queries = [] %}
{% for item in items %}
  {% if item.enabled %}
    {% do queries.append("select " ~ item.name) %}
  {% endif %}
{% endfor %}
{% if execute and queries | length > 0 %}
  {% set result = run_query(queries[0]) %}
{% elif queries | length > 0 %}
  {{ return(queries) }}
{% else %}
  {{ return([]) }}
{% endif %}
{% endmacro %}

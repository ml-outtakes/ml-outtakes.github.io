---
layout: main
---

{% for article in site.articles %}
<div class="card">
  <div class="card-content">
    <div class="content">
      <h2><a href="{{article.url}}">{{article.title}}</a></h2>
      <div>{{article.abstract | markdownify}}</div>
    </div>
  </div>
</div>
{% endfor %}

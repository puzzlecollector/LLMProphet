from django.db import models

# Create your models here.
class Like(models.Model):
    count = models.PositiveIntegerField(default=0)

class Visitor(models.Model):
    count = models.PositiveIntegerField(default=0)
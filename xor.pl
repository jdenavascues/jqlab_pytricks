#!/usr/bin/perl -w
use MIME::Base64 qw/encode_base64 decode_base64/;

if($#ARGV!=2) {
  print "Usage: \n";
  print " xor.pl -encode key secret\n";
  print " xor.pl -decode key code\n";
  print "Note: key and secret should be the same length. Do not reuse keys\n";
}

if($ARGV[0] eq "-encode") {
  $key = $ARGV[1];
  $secret = $ARGV[2];

  if(length($key) != length($secret)) {
    print "Key and secret lengths don't match :(\n";
    exit 1;
  }
  print "Code is: ". encode_base64($key ^ $secret) ."\n";
}

if($ARGV[0] eq "-decode") {
  $key = $ARGV[1];
  $code = decode_base64($ARGV[2]);
  if(length($key) != length($code)) {
    print "Key and code lengths don't match :(\n";
    exit 1;
  }
  print "Secret is: ".($key ^ $code)."\n";
}
